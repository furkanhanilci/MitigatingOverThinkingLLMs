"""
NUM_GPUS=8  # Total number of GPUs available
BATCH_SIZE=4  # Adjust batch size if needed
MODEL_NAME="qwen1.5B"

for RANK in $(seq 0 $((NUM_GPUS - 1))); do
    CUDA_VISIBLE_DEVICES=$RANK python tools/run_generation.py \
        --rank=$RANK \
        --world_size=$NUM_GPUS \
        --batch_size=$BATCH_SIZE \
        --dataset_name=math500 \
        --save_name=$MODEL_NAME \
        --model_name=$MODEL_NAME &
done

wait
"""
import copy
import json
import os

import torch
from absl import app, flags
from tqdm.auto import tqdm as auto_tqdm
from vllm import LLM, SamplingParams

from tools.eval_utils import (
    DATASET_TO_SAMPLING_PARAMS,
    load_aime2024,
    load_aime2223,
    load_local_amc23,
    load_local_olympiad_bench,
    load_math500,
)

flags.DEFINE_integer("rank", None, help="The rank for current progress.")
flags.DEFINE_integer("world_size", None, help="Total number of GPUs available.")
flags.DEFINE_integer("batch_size", 4, help="The generation batch size.")
flags.DEFINE_string(
    "model_name",
    "qwen1.5B",
    help="The evaluation model."
)
flags.DEFINE_integer(
    "tp_size",
    1,
    help="tensor parallel size of vLLM."
)
flags.DEFINE_string(
    "save_name",
    None,
    required=False,
    help="where to save the generation results."
)
flags.DEFINE_enum(
    "dataset_name",
    "math500",
    enum_values=[
        "math500",
        "aime",
        "amc23",
        "olympiad_bench",
        "aime2223",
    ],
    help="The evaluation dataset."
)

FLAGS = flags.FLAGS

def main(argv):
    rank = FLAGS.rank
    world_size = FLAGS.world_size
    batch_size = FLAGS.batch_size
    dataset_name = FLAGS.dataset_name

    if dataset_name == "math500":
        local_ds = load_math500(rank, world_size)
    elif dataset_name == "aime":
        local_ds = load_aime2024(rank, world_size)
    elif dataset_name == "aime2223":
        local_ds = load_aime2223(rank, world_size)
    elif dataset_name == "amc23":
        local_ds = load_local_amc23(rank, world_size)
    elif dataset_name == "olympiad_bench":
        local_ds = load_local_olympiad_bench(rank, world_size)
    else:
        raise ValueError()
    if FLAGS.model_name == "qwen1.5B":
        hf_model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    elif FLAGS.model_name == "qwen7B":
        hf_model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    elif FLAGS.model_name == "qwen1.5B-nonR1-instruct":
        hf_model_name = "Qwen/Qwen2.5-Math-1.5B-Instruct"
    elif FLAGS.model_name == "deepscaler":
        hf_model_name = "agentica-org/DeepScaleR-1.5B-Preview"
    else:
        hf_model_name = FLAGS.model_name

    if "qwq" in hf_model_name.lower():
        instruction = "Let's think step by step and output the final answer within \\boxed{}."
    else:
        instruction = "Please reason step by step, and put your final answer within \\boxed{}."
    print(f"Instruction that will be added to the input:\n{instruction}")

    sampling_params = DATASET_TO_SAMPLING_PARAMS[dataset_name]
    max_model_len = 32_768

    if FLAGS.model_name in ["qwen1.5B-nonR1-instruct"]:
        model = LLM(hf_model_name,dtype=torch.bfloat16)
        sampling_params["max_tokens"] = 4096
    else:
        model = LLM(
            hf_model_name,
            dtype=torch.bfloat16,
            max_model_len=max_model_len,
            tensor_parallel_size=FLAGS.tp_size,
            disable_custom_all_reduce=True,
        )

    if "qwq" in hf_model_name.lower():
        sampling_params["n"] = 16
        print("Only sample 16 responses for QwQ due to the large size")

    sampling_params = SamplingParams(
        temperature=sampling_params["temperature"],
        top_p=sampling_params["top_p"],
        n=sampling_params["n"],
        max_tokens=sampling_params["max_tokens"],
    )

    output_logs = []
    prog_bar = auto_tqdm(range(len(local_ds)))

    batched_messages = []
    for data_idx in range(len(local_ds)):
        question = local_ds[data_idx]["question"]
        if FLAGS.model_name in ["qwen1.5B-nonR1-instruct"]:
            message = [
                {
                    "role": "system",
                    "content": (
                        "Please reason step by step, and"
                        " put your final answer within \\boxed{}."
                    ),
                },
                {"role": "user", "content": question},
            ]
        else:
            message = [{"role": "user", "content": question + " " + instruction}]
        batched_messages.append(message)
        if len(batched_messages) == batch_size or data_idx == len(local_ds) - 1:
            outputs = model.chat(
                messages=batched_messages,
                add_generation_prompt=True,
                sampling_params=sampling_params,
                use_tqdm=False,
            )
            answers = [
                [
                    output.outputs[x].text for x in range(len(output.outputs))
                ]
                for output in outputs
            ]
            answer_token_ids_list = [
                [
                    output.outputs[x].token_ids for x in range(len(output.outputs))
                ]
                for output in outputs
                # output.outputs[0].token_ids for output in outputs
            ]

            # record logs
            for idx in range(len(answers)):
                curr_data_idx = data_idx - len(answers) + idx + 1
                exec_log = copy.deepcopy(local_ds[curr_data_idx])
                exec_log["model_answer"] = answers[idx]
                answer_tokens_ids = answer_token_ids_list[idx]
                exec_log["model_answer_length"] = [len(x) for x in answer_tokens_ids]

                output_logs.append(exec_log)

            batched_messages.clear()
        prog_bar.update(1)

    save_path = f"logs/{dataset_name}/{FLAGS.save_name}/rank{rank}.jsonl"
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    with open(save_path, "w") as f:
        for item in output_logs:
            f.write(json.dumps(item) + "\n")

if __name__ == "__main__":
    app.run(main)

