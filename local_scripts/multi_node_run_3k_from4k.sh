# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# if we detect a space character in the head node IP, we'll
# convert it to an ipv4 address. This step is optional.
if [[ "$head_node_ip" == *" "* ]]; then
IFS=' ' read -ra ADDR <<<"$head_node_ip"
if [[ ${#ADDR[0]} -gt 16 ]]; then
  head_node_ip=${ADDR[1]}
else
  head_node_ip=${ADDR[0]}
fi
echo "IPV6 address detected. We split the IPV4 address as $head_node_ip"
fi

port=6499
ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head"

echo "Starting HEAD at $head_node"

NUM_GPUS=$(nvidia-smi -L | wc -l)  # Count available GPUs
echo "SLURM_CPUS_PER_TASK: ${SLURM_CPUS_PER_TASK}"
echo "SLURM_GPUS_PER_TASK: ${SLURM_GPUS_PER_TASK}"
echo "SLURM_GPUS_PER_TASK: ${NUM_GPUS}"


cd /path/to/thinkprune
# Activate virtual environment
source ~/.bashrc
conda activate verl_thinkprune


export VLLM_ATTENTION_BACKEND=XFORMERS
export HYDRA_FULL_ERROR=1
export WANDB_API_KEY=YOUR_WANDB_KEY

srun rm -rf /tmp/ray/
# export RAY_TMPDIR=/path/to/ray_cache

LENGTH=3000
RUN_NAME=DeepSeek-R1-Distill-Qwen-1.5B-${LENGTH}-from4k
MODEL=checkpoints-merged/DeepSeek-R1-Distill-Qwen-1.5B-4000/global_step_180

N_GPUS=8
TP=1
MODEL_DIR=/path/to/checkpoints/${RUN_NAME}
DATA_DIR=/path/to/past_aime_amc/length${LENGTH}

BATCH_SIZE=64
ROLLOUT_BS=128
ROLLOUT_N=16


srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --node-ip-address="$head_node_ip" --port=$port  --block &

sleep 10

# Start worker nodes
worker_num=$((SLURM_JOB_NUM_NODES - 1))
for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        ray start --address "$ip_head" --block &
    sleep 5
done

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=${DATA_DIR}/train.parquet \
    data.val_files=${DATA_DIR}/test.parquet \
    data.train_batch_size=128 \
    data.val_batch_size=128 \
    data.max_prompt_length=768 \
    data.max_response_length=${LENGTH} \
    actor_rollout_ref.model.path=${MODEL} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size=$BATCH_SIZE \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=$ROLLOUT_BS \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TP \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    +actor_rollout_ref.rollout.disable_log_stats=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size=$ROLLOUT_BS \
    reward_model.enable=False \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.val_before_train=True \
    trainer.default_local_dir=$MODEL_DIR \
    trainer.default_hdfs_dir=null \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_math' \
    trainer.experiment_name=${RUN_NAME} \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=2 \
    trainer.multisample_val=True \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    trainer.total_epochs=10 \
    trainer.num_keep_checkpoint=20 \
    trainer.resume_checkpoint=True

