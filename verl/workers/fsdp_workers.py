# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The main entry point to run the PPO algorithm
"""

import os
import time
import warnings
import logging
import random
from typing import Dict, Any, List, Union
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from autotiktokenizer import AutoTikTokenizer

import torch.distributed
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    ShardedStateDictConfig,
    ShardedOptimStateDictConfig
)
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    FullStateDictConfig,
)
import torch
import torch.distributed
from torch.distributed.device_mesh import init_device_mesh
import torch.distributed.checkpoint as dist_cp
from torch.distributed.checkpoint.default_planner import (
    DefaultSavePlanner,
    DefaultLoadPlanner,
)
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

import verl.utils.hdfs_io as hdfs_io
import verl.utils.torch_functional as verl_F
from omegaconf import DictConfig, open_dict
from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import register, Dispatch
from verl.utils import hf_tokenizer
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.fsdp_utils import (
    get_fsdp_wrap_policy,
    offload_fsdp_grad,
    init_fn,
    get_init_weight_context_manager,
)
from verl.utils.fsdp_utils import (
    offload_fsdp_optimizer,
    offload_fsdp_param_and_grad,
    load_fsdp_optimizer,
    load_fsdp_param_and_grad,
)
from verl.utils.import_utils import import_external_libs
from verl.utils.model import compute_position_id_with_mask
from verl.utils.flops_counter import FlopsCounter
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager
from verl.trainer.ppo.trainer_utils import find_latest_checkpoint
from verl.utils.reward_score.qwen_math import extract_answer, math_equal

from codetiming import Timer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_PPO_LOGGING_LEVEL", "WARN"))


def create_device_mesh(world_size, fsdp_size):
    if fsdp_size < 0 or fsdp_size >= world_size:
        device_mesh = init_device_mesh(
            "cuda", mesh_shape=(world_size,), mesh_dim_names=["fsdp"]
        )
    else:
        raise ValueError(
            "HSDP is not supported yet because it produces incorrect results for now. Please set fsdp_size=-1"
        )
        assert world_size % fsdp_size == 0
        device_mesh = init_device_mesh(
            "cuda",
            mesh_shape=(world_size // fsdp_size, fsdp_size),
            mesh_dim_names=["ddp", "fsdp"],
        )
    return device_mesh


def get_sharding_strategy(device_mesh):
    from torch.distributed.fsdp import ShardingStrategy

    if device_mesh.ndim == 1:
        sharding_strategy = ShardingStrategy.FULL_SHARD
    elif device_mesh.ndim == 2:
        sharding_strategy = ShardingStrategy.HYBRID_SHARD
    else:
        raise NotImplementedError(
            f"Get device mesh ndim={device_mesh.ndim}, but only support 1 or 2"
        )
    return sharding_strategy


class ActorRolloutRefWorker(Worker):
    """
    This worker can be instantiated as a standalone actor or a standalone rollout or a standalone reference policy
    or a hybrid engine based on the config.rollout
    """

    def __init__(self, config: DictConfig, role: str):
        super().__init__()
        self.config = config
        import torch.distributed

        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")

        # build device mesh for FSDP
        world_size = torch.distributed.get_world_size()
        # TODO(sgm): support FSDP hybrid shard for larger model
        self.device_mesh = create_device_mesh(
            world_size=world_size, fsdp_size=self.config.actor.fsdp_config.fsdp_size
        )

        # build device mesh for Ulysses Sequence Parallel
        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.config.actor.get(
            "ulysses_sequence_parallel_size", 1
        )
        dp = world_size // self.ulysses_sequence_parallel_size
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh(
                "cuda",
                mesh_shape=(dp, self.ulysses_sequence_parallel_size),
                mesh_dim_names=["dp", "sp"],
            )

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(
            self.ulysses_device_mesh
        )

        self.role = role
        assert self.role in [
            "actor",
            "rollout",
            "ref",
            "actor_rollout",
            "actor_rollout_ref",
        ]

        self._is_actor = self.role in ["actor", "actor_rollout", "actor_rollout_ref"]
        self._is_rollout = self.role in [
            "rollout",
            "actor_rollout",
            "actor_rollout_ref",
        ]
        self._is_ref = self.role in ["ref", "actor_rollout_ref"]

        self._is_offload_param = False
        self._is_offload_grad = False
        self._is_offload_optimizer = False
        if self._is_actor:
            self._is_offload_param = self.config.actor.fsdp_config.get(
                "param_offload", False
            )
            self._is_offload_grad = self.config.actor.fsdp_config.get(
                "grad_offload", False
            )
            self._is_offload_optimizer = self.config.actor.fsdp_config.get(
                "optimizer_offload", False
            )
        elif self._is_ref:
            # TODO: it seems that manual offload is slowly than FSDP offload
            self._is_offload_param = self.config.ref.fsdp_config.get(
                "param_offload", False
            )

        # normalize config
        if self._is_actor:
            self.config.actor.ppo_mini_batch_size //= (
                self.device_mesh.shape[0] // self.ulysses_sequence_parallel_size
            )
            self.config.actor.ppo_micro_batch_size //= (
                self.device_mesh.shape[0] // self.ulysses_sequence_parallel_size
            )
            self.config.actor.ppo_mini_batch_size *= self.config.rollout.n
            # Bairu: delete this following the latest change in verl
            # self.config.actor.ppo_micro_batch_size *= self.config.rollout.n
        if self._is_rollout:
            self.config.rollout.log_prob_micro_batch_size //= (
                self.device_mesh.shape[0] // self.ulysses_sequence_parallel_size
            )
            # Bairu: delete this following the latest change in verl
            # self.config.rollout.log_prob_micro_batch_size *= self.config.rollout.n
        if self._is_ref:
            self.config.ref.log_prob_micro_batch_size //= (
                self.device_mesh.shape[0] // self.ulysses_sequence_parallel_size
            )
            # Bairu: delete this following the latest change in verl
            # self.config.ref.log_prob_micro_batch_size *= self.config.rollout.n

    def _build_model_optimizer(
        self,
        model_path,
        fsdp_config,
        optim_config,
        override_model_config,
        use_remove_padding=False,
        enable_gradient_checkpointing=False,
        trust_remote_code=False,
        role="actor",
    ):
        from verl.utils.model import print_model_size, update_model_config
        from verl.utils.torch_dtypes import PrecisionType
        from transformers import AutoModelForCausalLM, AutoConfig
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            ShardingStrategy,
            MixedPrecision,
            CPUOffload,
        )
        from torch import optim

        assert role in ["actor", "ref"]

        log_gpu_memory_usage("Before init from HF AutoModel", logger=logger)
        local_path = copy_local_path_from_hdfs(model_path)

        # note that we have to create model in fp32. Otherwise, the optimizer is in bf16, which is incorrect
        # TODO(zhangchi.usc1992): 1. support create from random initialized model. 2. Support init with FSDP directly
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)

        torch_dtype = fsdp_config.get("model_dtype", None)
        if torch_dtype is None:
            torch_dtype = torch.float32 if self._is_actor else torch.bfloat16
        else:
            torch_dtype = PrecisionType.to_dtype(torch_dtype)

        # override model kwargs
        actor_model_config = AutoConfig.from_pretrained(
            local_path, trust_remote_code=trust_remote_code
        )

        if use_remove_padding:
            from verl.models.registry import check_model_support_rmpad

            check_model_support_rmpad(actor_model_config.model_type)

        if use_remove_padding and self.ulysses_sequence_parallel_size > 1:
            from verl.models.transformers.monkey_patch import apply_monkey_patch

            apply_monkey_patch(actor_model_config, verbose=True)

        override_config_kwargs = {
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(override_model_config)
        update_model_config(
            actor_model_config, override_config_kwargs=override_config_kwargs
        )
        if self.rank == 0:
            print(f"Model config after override: {actor_model_config}")

        # NOTE(fix me): tie_word_embedding causes meta_tensor init to hang
        init_context = get_init_weight_context_manager(
            use_meta_tensor=not actor_model_config.tie_word_embeddings
        )

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            print("Loading actor model from ", local_path)
            actor_module = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=local_path,
                torch_dtype=torch_dtype,
                config=actor_model_config,
                attn_implementation="flash_attention_2",
                trust_remote_code=trust_remote_code,
            )
            # some parameters may not in torch_dtype. TODO(zhangchi.usc1992) remove this after we switch to fsdp2
            actor_module.to(torch_dtype)

            if enable_gradient_checkpointing:
                actor_module.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
        torch.distributed.barrier()

        if self.rank == 0:
            print_model_size(actor_module)

        log_gpu_memory_usage("After init from HF AutoModel", logger=logger)

        # We wrap FSDP for rollout as well
        mixed_precision_config = fsdp_config.get("mixed_precision", None)
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(
                mixed_precision_config.get("param_dtype", "bf16")
            )
            reduce_dtype = PrecisionType.to_dtype(
                mixed_precision_config.get("reduce_dtype", "fp32")
            )
            buffer_dtype = PrecisionType.to_dtype(
                mixed_precision_config.get("buffer_dtype", "fp32")
            )
        else:
            param_dtype = torch.bfloat16
            reduce_dtype = torch.float32
            buffer_dtype = torch.float32

        mixed_precision = MixedPrecision(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            buffer_dtype=buffer_dtype,
        )

        auto_wrap_policy = get_fsdp_wrap_policy(
            module=actor_module, config=fsdp_config.get("wrap_policy", None)
        )

        if self._is_rollout and self.config.rollout.name == "hf":
            # TODO(zhangchi.usc1992, shengguangming) fix me. Current, auto_wrap_policy causes HFRollout to hang in Gemma
            auto_wrap_policy = None

        print(f"wrap_policy: {auto_wrap_policy}")

        fsdp_mesh = self.device_mesh
        sharding_strategy = get_sharding_strategy(fsdp_mesh)

        # TODO: add transformer policy
        # We force reference policy to use CPUOffload to save memory.
        # We force turn off CPUOffload for actor because it causes incorrect results when using grad accumulation
        cpu_offload = None if role == "actor" else CPUOffload(offload_params=True)
        actor_module_fsdp = FSDP(
            actor_module,
            cpu_offload=cpu_offload,
            param_init_fn=init_fn,
            use_orig_params=False,
            auto_wrap_policy=auto_wrap_policy,
            device_id=torch.cuda.current_device(),
            sharding_strategy=sharding_strategy,  # zero3
            mixed_precision=mixed_precision,
            sync_module_states=True,
            device_mesh=self.device_mesh,
            forward_prefetch=False,
        )

        log_gpu_memory_usage("After Actor FSDP init", logger=logger)

        # TODO: add more optimizer args into config
        if role == "actor":
            from verl.utils.torch_functional import get_constant_schedule_with_warmup

            actor_optimizer = optim.AdamW(
                actor_module_fsdp.parameters(),
                lr=optim_config.lr,
                betas=optim_config.get("betas", (0.9, 0.999)),
                weight_decay=optim_config.get("weight_decay", 1e-2),
            )
            print("Initialized optimizer len", len(actor_optimizer.param_groups))

            total_steps = optim_config.get("total_training_steps", 0)
            num_warmup_steps_ratio = optim_config.get("lr_warmup_steps_ratio", 0.0)
            num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

            print(f"Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}")

            actor_lr_scheduler = get_constant_schedule_with_warmup(
                optimizer=actor_optimizer, num_warmup_steps=num_warmup_steps
            )
        else:
            actor_optimizer = None
            actor_lr_scheduler = None

        log_gpu_memory_usage("After actor optimizer init", logger=logger)

        return (
            actor_module_fsdp,
            actor_optimizer,
            actor_lr_scheduler,
            actor_model_config,
        )

    def _build_rollout(self):
        from torch.distributed.device_mesh import init_device_mesh

        # TODO(sgm): support FSDP hybrid shard for larger model
        infer_tp = self.config.rollout.tensor_model_parallel_size
        dp = self.world_size // infer_tp
        assert (
            self.world_size % infer_tp == 0
        ), f"rollout world_size: {self.world_size} is not divisible by infer_tp: {infer_tp}"
        rollout_device_mesh = init_device_mesh(
            "cuda", mesh_shape=(dp, infer_tp), mesh_dim_names=["dp", "infer_tp"]
        )

        if self.config.rollout.name == "hf":
            from verl.workers.rollout import HFRollout
            from verl.workers.sharding_manager import BaseShardingManager

            rollout = HFRollout(
                module=self.actor_module_fsdp, config=self.config.rollout
            )
            rollout_sharding_manager = BaseShardingManager()
            # TODO: a sharding manager that do nothing?
        elif self.config.rollout.name == "vllm":
            from verl.workers.rollout.vllm_rollout import vLLMRollout
            from verl.workers.sharding_manager import FSDPVLLMShardingManager

            log_gpu_memory_usage("Before building vllm rollout", logger=None)
            rollout = vLLMRollout(
                actor_module=self.actor_module_fsdp,
                config=self.config.rollout,
                tokenizer=self.tokenizer,
                model_hf_config=self.actor_model_config,
            )
            log_gpu_memory_usage("After building vllm rollout", logger=None)
            if torch.distributed.get_world_size() == 1:
                self.config.rollout.load_format = "dummy_hf"
            rollout_sharding_manager = FSDPVLLMShardingManager(
                module=self.actor_module_fsdp,
                inference_engine=rollout.inference_engine,
                model_config=self.actor_model_config,
                full_params="hf" in self.config.rollout.load_format,
                device_mesh=rollout_device_mesh,
            )
            log_gpu_memory_usage("After building sharding manager", logger=None)

        return rollout, rollout_sharding_manager

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        from verl.workers.actor import DataParallelPPOActor

        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get("external_lib", None))

        from omegaconf import OmegaConf

        override_model_config = OmegaConf.to_container(
            self.config.model.get("override_config", OmegaConf.create())
        )

        use_remove_padding = self.config.model.get("use_remove_padding", False)

        if self._is_actor or self._is_rollout:
            # we need the model for actor and rollout
            if self._is_actor:
                optim_config = self.config.actor.optim
                fsdp_config = self.config.actor.fsdp_config
            else:
                # Bairu: follow DeepScalR to fix the bug in main generation
                # optim_config = None
                optim_config = self.config.actor.optim
                fsdp_config = OmegaConf.create()

            if self.config.model.checkpoint_path is None:
                to_load_path = self.config.model.path
            else:
                # find latest checkpoint
                latest_checkpint = find_latest_checkpoint(
                    self.config.model.checkpoint_path
                )
                if latest_checkpint is None:
                    to_load_path = self.config.model.path
                else:
                    to_load_path = latest_checkpint

            (
                self.actor_module_fsdp,
                self.actor_optimizer,
                self.actor_lr_scheduler,
                self.actor_model_config,
            ) = self._build_model_optimizer(
                # model_path=to_load_path,
                model_path=self.config.model.path,
                fsdp_config=fsdp_config,
                optim_config=optim_config,
                override_model_config=override_model_config,
                use_remove_padding=use_remove_padding,
                enable_gradient_checkpointing=self.config.model.get(
                    "enable_gradient_checkpointing", False
                ),
                trust_remote_code=self.config.model.get("trust_remote_code", False),
                role="actor",
            )

            if self.config.model.checkpoint_path is not None:
                # overwrite optimzier states
                if to_load_path != self.config.model.path:
                    local_model_path = os.path.join(
                        to_load_path, f'model_world_size_{self.world_size}_rank_{self.rank}.pt',
                    )
                    local_optim_path = os.path.join(
                        to_load_path, f'optim_world_size_{self.world_size}_rank_{self.rank}.pt',
                    )
                    local_extra_state_path = os.path.join(
                        to_load_path, f'extra_state_world_size_{self.world_size}_rank_{self.rank}.pt',
                    )
                    print(
                        f"[rank-{self.rank}]: Loading from {local_model_path} "
                        f"and {local_optim_path} and {local_extra_state_path}"
                    )
                    model_state_dict = torch.load(local_model_path)
                    optimizer_state_dict = torch.load(local_optim_path)
                    extra_state_dict = torch.load(local_extra_state_path)

                    lr_scheduler_state_dict = extra_state_dict['lr_scheduler']

                    state_dict_cfg = ShardedStateDictConfig(offload_to_cpu=True)
                    optim_cfg = ShardedOptimStateDictConfig(offload_to_cpu=True)
                    with FSDP.state_dict_type(self.actor_module_fsdp, StateDictType.SHARDED_STATE_DICT, state_dict_cfg, optim_cfg):
                        self.actor_module_fsdp.load_state_dict(model_state_dict)
                        self.actor_optimizer.load_state_dict(optimizer_state_dict)

                    self.actor_lr_scheduler.load_state_dict(lr_scheduler_state_dict)

            # get the original unwrapped module
            self.actor_module = self.actor_module_fsdp._fsdp_wrapped_module

            if self._is_offload_param:
                # param is require during state_dict in sharding manager
                offload_fsdp_grad(module=self.actor_module_fsdp)
                log_gpu_memory_usage(
                    "After offload actor grad during init", logger=logger
                )
            if self._is_offload_optimizer:
                offload_fsdp_optimizer(optimizer=self.actor_optimizer)
                log_gpu_memory_usage(
                    "After offload actor optimizer during init", logger=logger
                )

        # load from checkpoint
        if self._is_actor:
            OmegaConf.set_struct(self.config.actor, True)
            with open_dict(self.config.actor):
                self.config.actor.use_remove_padding = use_remove_padding
            self.actor = DataParallelPPOActor(
                config=self.config.actor,
                actor_module=self.actor_module_fsdp,
                actor_optimizer=self.actor_optimizer,
            )

        if self._is_rollout:
            self.rollout, self.rollout_sharding_manager = self._build_rollout()

        if self._is_ref:
            self.ref_module_fsdp = self._build_model_optimizer(
                model_path=self.config.model.path,
                fsdp_config=self.config.ref.fsdp_config,
                optim_config=None,
                override_model_config=override_model_config,
                use_remove_padding=use_remove_padding,
                trust_remote_code=self.config.model.get("trust_remote_code", False),
                role="ref",
            )[0]
            OmegaConf.set_struct(self.config.ref, True)
            with open_dict(self.config.ref):
                self.config.ref.use_remove_padding = use_remove_padding
            self.ref_policy = DataParallelPPOActor(
                config=self.config.ref, actor_module=self.ref_module_fsdp
            )

        if self._is_actor:
            self.flops_counter = FlopsCounter(self.actor_model_config)

        torch.cuda.empty_cache()

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_actor(self, data: DataProto):
        data = data.to("cuda")

        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_param_and_grad(
                module=self.actor_module_fsdp,
                device_id=torch.cuda.current_device(),
                load_grad=self._is_offload_grad,
            )
        if self._is_offload_optimizer:
            load_fsdp_optimizer(
                optimizer=self.actor_optimizer, device_id=torch.cuda.current_device()
            )

        data.batch = data.batch.cuda()

        log_gpu_memory_usage("Before update policy", logger=logger)

        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)
            # perform training
            with Timer(name="update_policy", logger=None) as timer:
                metrics = self.actor.update_policy(data=data)
            delta_time = timer.last
            global_num_tokens = data.meta_info["global_token_num"]
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(
                global_num_tokens, delta_time
            )
            metrics["mfu/actor"] = (
                estimated_flops
                * self.config.actor.ppo_epochs
                / promised_flops
                / self.world_size
            )

            self.actor_lr_scheduler.step()
            lr = self.actor_lr_scheduler.get_last_lr()[0]
            metrics["actor/lr"] = lr

            log_gpu_memory_usage("After update policy", logger=logger)

            # TODO: here, we should return all metrics
            output = DataProto(meta_info={"metrics": metrics})

            output = self.ulysses_sharding_manager.postprocess_data(data=output)
            output = output.to("cpu")

        if self._is_offload_param:
            offload_fsdp_param_and_grad(
                module=self.actor_module_fsdp, offload_grad=self._is_offload_grad
            )
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.actor_optimizer)
        torch.cuda.empty_cache()
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_sequences(self, prompts: DataProto):
        prompts = prompts.to("cuda")
        # set to False if it is validation
        recompute_log_prob = prompts.meta_info.get("recompute_log_prob", True)

        assert self._is_rollout
        if self._is_offload_param:
            load_fsdp_param_and_grad(
                module=self.actor_module_fsdp,
                device_id=torch.cuda.current_device(),
                load_grad=self._is_offload_grad,
            )

        prompts.batch = prompts.batch.cuda()
        meta_info = {
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        prompts.meta_info.update(meta_info)
        with self.rollout_sharding_manager:
            log_gpu_memory_usage(
                "After entering rollout sharding manager", logger=logger
            )

            prompts = self.rollout_sharding_manager.preprocess_data(prompts)
            output = self.rollout.generate_sequences(prompts=prompts)

            log_gpu_memory_usage("After rollout generation", logger=logger)

            output = self.rollout_sharding_manager.postprocess_data(output)

        output = output.to("cpu")

        if self._is_offload_param:
            # NOTE(sgm): the grad is already in CPU, only offload param here
            offload_fsdp_param_and_grad(
                module=self.actor_module_fsdp, offload_grad=self._is_offload_grad
            )
        # clear kv cache
        torch.cuda.empty_cache()
        log_gpu_memory_usage("After recompute log prob", logger=logger)
        return output

    # @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO, blocking=False)
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_log_prob(self, data: DataProto):
        assert self._is_actor
        data = data.to("cuda")
        # we should always recompute old_log_probs when it is HybridEngine
        data.meta_info["micro_batch_size"] = (
            self.config.rollout.log_prob_micro_batch_size
        )
        data.meta_info["max_token_len"] = (
            self.config.rollout.log_prob_max_token_len_per_gpu
        )
        data.meta_info["use_dynamic_bsz"] = self.config.rollout.log_prob_use_dynamic_bsz
        data.meta_info["temperature"] = self.config.rollout.temperature
        # perform recompute log_prob
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data)
            output = self.actor.compute_log_prob(data=data)
            output = DataProto.from_dict(
                tensors={"old_log_probs": output},
                meta_info={"temperature": self.config.rollout.temperature},
            )
            output = self.ulysses_sharding_manager.postprocess_data(output)

        output = output.to("cpu")

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        if self.world_size > 1:
            self.actor.actor_module._handle.reshard(True)

        torch.cuda.empty_cache()
        return output

    # @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO, blocking=False)
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_ref_log_prob(self, data: DataProto):
        assert self._is_ref

        data = data.to("cuda")

        micro_batch_size = self.config.ref.log_prob_micro_batch_size
        data.meta_info["micro_batch_size"] = micro_batch_size
        data.meta_info["temperature"] = self.config.rollout.temperature
        data.meta_info["max_token_len"] = self.config.ref.log_prob_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.ref.log_prob_use_dynamic_bsz
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data)
            output = self.ref_policy.compute_log_prob(data=data)
            output = DataProto.from_dict(tensors={"ref_log_prob": output})
            output = self.ulysses_sharding_manager.postprocess_data(output)

        output = output.to("cpu")

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        if self.world_size > 1:
            self.ref_policy.actor_module._handle.reshard(True)

        torch.cuda.empty_cache()
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None):
        assert self._is_actor
        import torch
        if self.rank == 0:
            if not os.path.exists(local_path):
                os.makedirs(local_path, exist_ok=False)
        torch.distributed.barrier()

        if self._is_offload_param:
            load_fsdp_param_and_grad(
                module=self.actor_module_fsdp,
                device_id=torch.cuda.current_device(),
                load_grad=self._is_offload_grad,
            )
        if self._is_offload_optimizer:
            load_fsdp_optimizer(
                self.actor_optimizer,
                device_id=torch.cuda.current_device(),
            )

        # TODO: support DCP and save sharded checkpoints

        state_dict_cfg = ShardedStateDictConfig(offload_to_cpu=True)
        optim_cfg = ShardedOptimStateDictConfig(offload_to_cpu=True)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with FSDP.state_dict_type(self.actor.actor_module, StateDictType.SHARDED_STATE_DICT, state_dict_cfg, optim_cfg):
                model_state_dict = self.actor.actor_module.state_dict()
                optimizer_state_dict = self.actor_optimizer.state_dict()
                lr_scheduler_state_dict = self.actor_lr_scheduler.state_dict()

                extra_state_dict = {
                    'lr_scheduler': lr_scheduler_state_dict,
                }
                model_path = os.path.join(local_path, f'model_world_size_{self.world_size}_rank_{self.rank}.pt')
                optim_path = os.path.join(local_path, f'optim_world_size_{self.world_size}_rank_{self.rank}.pt')
                extra_path = os.path.join(local_path, f'extra_state_world_size_{self.world_size}_rank_{self.rank}.pt')

                print(f'[rank-{self.rank}]: Saving model to {os.path.abspath(model_path)}')
                print(f'[rank-{self.rank}]: Saving checkpoint to {os.path.abspath(model_path)}')
                print(f'[rank-{self.rank}]: Saving extra_state to {os.path.abspath(extra_path)}')
                torch.save(model_state_dict, model_path)
                torch.save(optimizer_state_dict, optim_path)  # TODO: address optimizer is None
                torch.save(extra_state_dict, extra_path)

        # wait for everyone to dump to local
        torch.distributed.barrier()

        if self.rank == 0:
            hf_local_path = os.path.join(local_path, 'huggingface')
            os.makedirs(hf_local_path, exist_ok=True)
            self.actor.actor_module._fsdp_wrapped_module.config.save_pretrained(hf_local_path)
            self.tokenizer.save_pretrained(local_path)

        torch.distributed.barrier()


class CriticWorker(Worker):
    def __init__(self, config):
        super().__init__()
        import torch.distributed

        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
        self.config = config

        # build device mesh for Ulysses Sequence Parallel
        world_size = torch.distributed.get_world_size()
        from torch.distributed.device_mesh import init_device_mesh

        fsdp_size = self.config.model.fsdp_config.fsdp_size
        self.device_mesh = create_device_mesh(
            world_size=world_size, fsdp_size=fsdp_size
        )

        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.config.get(
            "ulysses_sequence_parallel_size", 1
        )
        dp = world_size // self.ulysses_sequence_parallel_size
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh(
                "cuda",
                mesh_shape=(dp, self.ulysses_sequence_parallel_size),
                mesh_dim_names=["dp", "sp"],
            )

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(
            self.ulysses_device_mesh
        )

        # set FSDP offload params
        self._is_offload_param = self.config.model.fsdp_config.param_offload
        self._is_offload_grad = self.config.model.fsdp_config.grad_offload
        self._is_offload_optimizer = self.config.model.fsdp_config.optimizer_offload

        # normalize config
        self.config.ppo_mini_batch_size //= (
            torch.distributed.get_world_size() // self.ulysses_sequence_parallel_size
        )
        self.config.ppo_micro_batch_size //= (
            torch.distributed.get_world_size() // self.ulysses_sequence_parallel_size
        )
        self.config.forward_micro_batch_size //= (
            torch.distributed.get_world_size() // self.ulysses_sequence_parallel_size
        )

    def _build_critic_model_optimizer(self, config):
        # the following line is necessary
        from verl.utils.model import LambdaLayer, print_model_size, squeeze
        from verl.utils.torch_dtypes import PrecisionType
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            ShardingStrategy,
            MixedPrecision,
        )
        from torch import optim

        local_path = copy_local_path_from_hdfs(config.model.path)
        # note that the tokenizer between actor and critic may be different. So override tokenizer info with actor info
        # using random initialized model from any architecture. May not be the same as Actor.

        tokenizer_path = copy_local_path_from_hdfs(config.model.tokenizer_path)
        self.tokenizer = hf_tokenizer(
            tokenizer_path,
            trust_remote_code=config.model.get("trust_remote_code", False),
        )

        from omegaconf import OmegaConf

        override_config = OmegaConf.to_container(
            self.config.model.get("override_config", OmegaConf.create())
        )
        override_config_kwargs = {
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(override_config)
        if self.rank == 0:
            print(f"Critic overriding config {override_config_kwargs}")

        torch_dtype = self.config.model.fsdp_config.get("model_dtype", "fp32")
        torch_dtype = PrecisionType.to_dtype(torch_dtype)

        from transformers import AutoConfig, AutoModelForTokenClassification
        from torch import nn

        trust_remote_code = False
        critic_model_config = AutoConfig.from_pretrained(
            local_path, trust_remote_code=trust_remote_code
        )
        critic_model_config.num_labels = 1

        use_remove_padding = config.model.get("use_remove_padding", False)
        if use_remove_padding:
            from verl.models.registry import check_model_support_rmpad

            check_model_support_rmpad(critic_model_config.model_type)

        if use_remove_padding and self.ulysses_sequence_parallel_size > 1:
            from verl.models.transformers.monkey_patch import apply_monkey_patch

            apply_monkey_patch(critic_model_config, verbose=True)

        init_context = get_init_weight_context_manager()
        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            setattr(critic_model_config, "classifier_dropout", 0.0)
            setattr(critic_model_config, "hidden_dropout", "0")
            critic_module = AutoModelForTokenClassification.from_pretrained(
                pretrained_model_name_or_path=local_path,
                torch_dtype=torch_dtype,
                config=critic_model_config,
                attn_implementation="flash_attention_2",
                trust_remote_code=trust_remote_code,
            )

            # some parameters may not in torch_dtype
            critic_module.to(torch_dtype)

            if config.model.get("enable_gradient_checkpointing", False):
                critic_module.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
        if self.rank == 0:
            print_model_size(critic_module)

        self.critic_model_config = critic_model_config

        fsdp_config = self.config.model.fsdp_config
        mixed_precision_config = fsdp_config.get("mixed_precision", None)
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(
                mixed_precision_config.get("param_dtype", "bf16")
            )
            reduce_dtype = PrecisionType.to_dtype(
                mixed_precision_config.get("reduce_dtype", "fp32")
            )
            buffer_dtype = PrecisionType.to_dtype(
                mixed_precision_config.get("buffer_dtype", "fp32")
            )
        else:
            param_dtype = torch.bfloat16
            reduce_dtype = torch.float32
            buffer_dtype = torch.float32

        mixed_precision = MixedPrecision(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            buffer_dtype=buffer_dtype,
        )

        auto_wrap_policy = get_fsdp_wrap_policy(
            module=critic_module, config=self.config.model.fsdp_config.wrap_policy
        )

        log_gpu_memory_usage("Before critic FSDP", logger=None)

        fsdp_mesh = self.device_mesh
        sharding_strategy = get_sharding_strategy(fsdp_mesh)

        # Note: We force turn off CPUOffload for critic because it causes incorrect results when using grad accumulation
        critic_module = FSDP(
            critic_module,
            param_init_fn=init_fn,
            use_orig_params=False,
            auto_wrap_policy=auto_wrap_policy,
            device_id=torch.cuda.current_device(),
            sharding_strategy=sharding_strategy,
            mixed_precision=mixed_precision,
            sync_module_states=True,
            forward_prefetch=False,
            device_mesh=self.device_mesh,
            cpu_offload=None,
        )

        log_gpu_memory_usage("After critic FSDP", logger=None)

        critic_optimizer = optim.AdamW(
            critic_module.parameters(),
            lr=config.optim.lr,
            betas=config.optim.get("betas", (0.9, 0.999)),
            weight_decay=config.optim.get("weight_decay", 1e-2),
        )

        total_steps = config.optim.get("total_training_steps", 0)
        num_warmup_steps_ratio = config.optim.get("lr_warmup_steps_ratio", 0.0)
        num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

        print(f"Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}")

        from verl.utils.torch_functional import get_constant_schedule_with_warmup

        critic_lr_scheduler = get_constant_schedule_with_warmup(
            optimizer=critic_optimizer, num_warmup_steps=num_warmup_steps
        )

        return critic_module, critic_optimizer, critic_lr_scheduler

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get("external_lib", None))

        from verl.workers.critic import DataParallelPPOCritic, DataParallelStepPPOCritic

        self.critic_module, self.critic_optimizer, self.critic_lr_scheduler = (
            self._build_critic_model_optimizer(self.config)
        )

        if self._is_offload_param:
            offload_fsdp_param_and_grad(
                module=self.critic_module, offload_grad=self._is_offload_grad
            )
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.critic_optimizer)

        critic_cls = DataParallelStepPPOCritic if self.config.step_critic else DataParallelPPOCritic
        self.critic = critic_cls(
            config=self.config,
            critic_module=self.critic_module,
            critic_optimizer=self.critic_optimizer,
        )

        self.flops_counter = FlopsCounter(self.critic_model_config)

        torch.cuda.empty_cache()

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_values(self, data: DataProto):
        data = data.to("cuda")

        if self._is_offload_param:
            load_fsdp_param_and_grad(
                module=self.critic_module,
                device_id=torch.cuda.current_device(),
                load_grad=self._is_offload_grad,
            )
        micro_batch_size = self.config.forward_micro_batch_size
        data.meta_info["micro_batch_size"] = micro_batch_size
        data.meta_info["max_token_len"] = self.config.forward_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.use_dynamic_bsz
        # perform forward computation
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)
            values = self.critic.compute_values(data=data)
            output = DataProto.from_dict(tensors={"values": values})
            output = self.ulysses_sharding_manager.postprocess_data(data=output)

        output = output.to("cpu")
        if self._is_offload_param:
            offload_fsdp_param_and_grad(
                module=self.critic_module, offload_grad=self._is_offload_grad
            )
        torch.cuda.empty_cache()
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_critic(self, data: DataProto):
        data = data.to("cuda")
        if self._is_offload_param:
            load_fsdp_param_and_grad(
                module=self.critic_module,
                device_id=torch.cuda.current_device(),
                load_grad=self._is_offload_grad,
            )
        if self._is_offload_optimizer:
            load_fsdp_optimizer(
                optimizer=self.critic_optimizer, device_id=torch.cuda.current_device()
            )

        # perform forward computation
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)

            with Timer(name="update_critic", logger=None) as timer:
                metrics = self.critic.update_critic(data=data)
            delta_time = timer.last

            global_num_tokens = data.meta_info["global_token_num"]
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(
                global_num_tokens, delta_time
            )
            metrics["mfu/critic"] = (
                estimated_flops
                * self.config.ppo_epochs
                / promised_flops
                / self.world_size
            )

            self.critic_lr_scheduler.step()
            lr = self.critic_lr_scheduler.get_last_lr()[0]
            metrics["critic/lr"] = lr

            output = DataProto(batch=None, meta_info={"metrics": metrics})
            output = self.ulysses_sharding_manager.postprocess_data(data=output)

        if self._is_offload_param:
            offload_fsdp_param_and_grad(
                module=self.critic_module, offload_grad=self._is_offload_grad
            )
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.critic_optimizer)
        torch.cuda.empty_cache()
        output = output.to("cpu")
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None):
        import torch

        if self._is_offload_param:
            load_fsdp_param_and_grad(
                module=self.critic_module,
                device_id=torch.cuda.current_device(),
                load_grad=self._is_offload_grad,
            )

        # TODO: support DCP and save sharded checkpoints
        import torch.distributed
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            StateDictType,
            FullStateDictConfig,
        )

        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(
            self.critic_module, StateDictType.FULL_STATE_DICT, cfg
        ):
            state_dict = self.critic_module.state_dict()
            optimizer_state_dict = FSDP.full_optim_state_dict(
                self.critic_module, self.critic_optimizer
            )

        if self.rank == 0:
            print(f"Saving critic checkpoint to {local_path}")
            os.makedirs(local_path, exist_ok=True)
            self.critic_module._fsdp_wrapped_module.save_pretrained(
                local_path, state_dict=state_dict
            )
            self.tokenizer.save_pretrained(local_path)
            if hdfs_path is not None:
                print(f"Uploading critic checkpoint to {hdfs_path}")
                hdfs_io.makedirs(hdfs_path, exist_ok=True)
                hdfs_io.copy(src=local_path, dst=hdfs_path)

        print(f"Saving optimizer state checkpoint to {local_path}")
        t0 = time.perf_counter()
        # distributed_writer = dist_cp.FileSystemWriter(
        #     os.path.join(local_path, "optim.pt")
        # )
        dist_cp.save(
            state_dict=optimizer_state_dict,
            checkpoint_id=os.path.join(local_path, "optimizers"),
            # sotrage_writer=distributed_writer,
        )
        t1 = time.perf_counter()
        print(
            f"Optimizer state checkpoint saved to {local_path}; Time = {t1 - t0:.4f}\n"
        )

        torch.distributed.barrier()
        if self._is_offload_param:
            offload_fsdp_param_and_grad(
                module=self.critic_module, offload_grad=self._is_offload_grad
            )


# TODO(sgm): we may need to extract it to dp_reward_model.py
class RewardModelWorker(Worker):
    """
    Note that we only implement the reward model that is subclass of AutoModelForTokenClassification.
    """

    def __init__(self, config):
        super().__init__()
        import torch.distributed

        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
        self.config = config

        # build device mesh for Ulysses Sequence Parallel
        world_size = torch.distributed.get_world_size()
        from torch.distributed.device_mesh import init_device_mesh

        fsdp_size = self.config.model.fsdp_config.fsdp_size
        self.device_mesh = create_device_mesh(
            world_size=world_size, fsdp_size=fsdp_size
        )

        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.config.get(
            "ulysses_sequence_parallel_size", 1
        )
        dp = world_size // self.ulysses_sequence_parallel_size
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh(
                "cuda",
                mesh_shape=(dp, self.ulysses_sequence_parallel_size),
                mesh_dim_names=["dp", "sp"],
            )

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(
            self.ulysses_device_mesh
        )

        self.use_remove_padding = self.config.model.get("use_remove_padding", False)
        self.config.micro_batch_size //= torch.distributed.get_world_size()

    def _build_tokenizer(self, config, local_path):
        # May require to switch tokenizer
        if self.config.model.input_tokenizer is None:
            self._do_switch_chat_template = False
        else:
            self._do_switch_chat_template = True
            input_tokneizer_local_path = copy_local_path_from_hdfs(
                self.config.model.input_tokenizer
            )

            self.input_tokenizer = hf_tokenizer(
                input_tokneizer_local_path,
                trust_remote_code=config.model.get("trust_remote_code", False),
            )
            self.tokenizer = hf_tokenizer(
                local_path,
                trust_remote_code=config.model.get("trust_remote_code", False),
            )

    def _build_model(self, config):
        # the following line is necessary
        from transformers import AutoModelForTokenClassification, AutoConfig
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            ShardingStrategy,
            CPUOffload,
        )

        # download the checkpoint from hdfs
        local_path = copy_local_path_from_hdfs(config.model.path)
        self._build_tokenizer(config, local_path)

        trust_remote_code = config.model.get("trust_remote_code", False)
        model_config = AutoConfig.from_pretrained(
            local_path, trust_remote_code=trust_remote_code
        )
        model_config.num_labels = 1

        use_remove_padding = config.model.get("use_remove_padding", False)
        if use_remove_padding:
            from verl.models.registry import check_model_support_rmpad

            check_model_support_rmpad(model_config.model_type)

        if use_remove_padding and self.ulysses_sequence_parallel_size > 1:
            from verl.models.transformers.monkey_patch import apply_monkey_patch

            apply_monkey_patch(model_config, verbose=True)

        # note that we have to create model in fp32. Otherwise, the optimizer is in bf16, which is incorrect
        init_context = get_init_weight_context_manager(
            use_meta_tensor=not model_config.tie_word_embeddings
        )

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            setattr(model_config, "classifier_dropout", 0.0)
            reward_module = AutoModelForTokenClassification.from_pretrained(
                pretrained_model_name_or_path=local_path,
                config=model_config,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                trust_remote_code=trust_remote_code,
            )
            reward_module.to(torch.bfloat16)
        auto_wrap_policy = get_fsdp_wrap_policy(
            module=reward_module, config=self.config.model.fsdp_config
        )

        fsdp_mesh = self.device_mesh
        sharding_strategy = get_sharding_strategy(fsdp_mesh)

        reward_module = FSDP(
            reward_module,
            param_init_fn=init_fn,
            use_orig_params=False,
            auto_wrap_policy=auto_wrap_policy,
            device_id=torch.cuda.current_device(),
            sharding_strategy=sharding_strategy,  # zero3
            sync_module_states=True,
            cpu_offload=CPUOffload(offload_params=True),
            forward_prefetch=False,
            device_mesh=self.device_mesh,
        )

        return reward_module

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get("external_lib", None))
        self.reward_module = self._build_model(config=self.config)
        torch.cuda.empty_cache()

    def _forward_micro_batch(self, micro_batch):
        from flash_attn.bert_padding import (
            pad_input,
            unpad_input,
            index_first_axis,
            rearrange,
        )
        from verl.utils.ulysses import (
            ulysses_pad_and_slice_inputs,
            gather_outpus_and_unpad,
        )

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                position_ids_rmpad = index_first_axis(
                    rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                    indices,
                ).transpose(0, 1)

                # pad and slice the inputs if sp > 1
                if self.ulysses_sequence_parallel_size > 1:
                    input_ids_rmpad, position_ids_rmpad, pad_size = (
                        ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    )

                # only pass input_ids and position_ids to enable flash_attn_varlen
                output = self.reward_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    use_cache=False,
                )  # prevent model thinks we are generating
                reward_rmpad = output.logits
                reward_rmpad = reward_rmpad.squeeze(0)  # (total_nnz)

                # gather output if sp > 1
                if self.ulysses_sequence_parallel_size > 1:
                    reward_rmpad = gather_outpus_and_unpad(
                        reward_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size
                    )

                # pad it back
                rm_score = pad_input(
                    reward_rmpad, indices=indices, batch=batch_size, seqlen=seqlen
                ).squeeze(-1)
            else:
                output = self.reward_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )
                rm_score = output.logits  # (batch_size, seq_len, 1)
                rm_score = rm_score.squeeze(-1)

            # extract the result of the last valid token
            eos_mask_idx = torch.argmax(position_ids * attention_mask, dim=-1)  # (bsz,)
            rm_score = rm_score[torch.arange(batch_size), eos_mask_idx]
            return rm_score

    def _expand_to_token_level(self, data: DataProto, scores: torch.Tensor):
        batch_size = data.batch.batch_size[0]
        # expand as token_level_reward
        attention_mask = data.batch["attention_mask"]
        position_ids = data.batch["position_ids"]
        response_length = data.batch["responses"].shape[-1]
        eos_mask_idx = torch.argmax(position_ids * attention_mask, dim=-1)  # (bsz,)
        token_level_scores = torch.zeros_like(
            attention_mask, dtype=scores.dtype
        )  # (bsz, seqlen)
        token_level_scores[torch.arange(batch_size), eos_mask_idx] = scores

        # select the response part
        token_level_scores = token_level_scores[:, -response_length:]

        return token_level_scores

    def _switch_chat_template(self, data: DataProto):
        src_max_length = data.batch["attention_mask"].shape[-1]

        src_tokenizer = self.input_tokenizer
        target_tokenizer = self.tokenizer

        rm_input_ids = []
        rm_attention_mask = []

        for i in range(data.batch.batch_size[0]):
            # extract raw prompt
            chat: list = data.non_tensor_batch["raw_prompt"][i].tolist()

            # extract response
            response_ids = data.batch["responses"][i]
            response_length = response_ids.shape[-1]
            valid_response_length = data.batch["attention_mask"][i][
                -response_length:
            ].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            response = src_tokenizer.decode(valid_response_ids)
            # remove bos and eos
            response = response.replace(src_tokenizer.eos_token, "")

            chat.append({"role": "assistant", "content": response})

            prompt_with_chat_template = target_tokenizer.apply_chat_template(
                chat, add_generation_prompt=False, tokenize=False
            )
            if self.rank == 0 and i == 0:
                # for debugging purpose
                print(f"Switch template. chat: {prompt_with_chat_template}")

            # the maximum length is actually determined by the reward model itself
            max_length = self.config.get("max_length", src_max_length)
            if max_length is None:
                max_length = src_max_length
            input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
                prompt=prompt_with_chat_template,
                tokenizer=target_tokenizer,
                max_length=max_length,
                pad_token_id=target_tokenizer.pad_token_id,
                left_pad=False,  # right padding
                truncation=self.config.get("truncation", "right"),
            )  # truncate from the right

            rm_input_ids.append(input_ids)
            rm_attention_mask.append(attention_mask)

        rm_input_ids = torch.cat(rm_input_ids, dim=0)
        rm_attention_mask = torch.cat(rm_attention_mask, dim=0)

        rm_position_ids = compute_position_id_with_mask(rm_attention_mask)

        rm_inputs = {
            "input_ids": rm_input_ids,
            "attention_mask": rm_attention_mask,
            "position_ids": rm_position_ids,
        }

        return DataProto.from_dict(rm_inputs)

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_rm_score(self, data: DataProto):
        import itertools
        from verl.utils.seqlen_balancing import rearrange_micro_batches, get_reverse_idx

        data = data.to("cuda")
        if self._do_switch_chat_template:
            rm_data = self._switch_chat_template(data)

        rm_data.batch = rm_data.batch.cuda()

        # perform forward computation
        with self.ulysses_sharding_manager:
            rm_data = self.ulysses_sharding_manager.preprocess_data(data=rm_data)
            data = self.ulysses_sharding_manager.preprocess_data(data=data)

            use_dynamic_bsz = self.config.use_dynamic_bsz
            if use_dynamic_bsz:
                max_token_len = (
                    self.config.forward_max_token_len_per_gpu
                    * self.ulysses_sequence_parallel_size
                )
                micro_batches, indices = rearrange_micro_batches(
                    batch=rm_data.batch, max_token_len=max_token_len
                )
            else:
                micro_batches = rm_data.batch.split(self.config.micro_batch_size)
            output = []
            for micro_batch in micro_batches:
                rm_score = self._forward_micro_batch(micro_batch)
                output.append(rm_score)
            scores = torch.cat(output, dim=0)  # (batch_size)

            if use_dynamic_bsz:
                indices = list(itertools.chain.from_iterable(indices))
                assert len(indices) == scores.size(
                    0
                ), f"{len(indices)} vs. {scores.size()}"
                revert_indices = torch.tensor(
                    get_reverse_idx(indices), dtype=torch.long
                )
                scores = scores[revert_indices]

            token_level_scores = self._expand_to_token_level(data, scores)
            # Note that this is only the scores, may not be the final rewards used to train RL
            output = DataProto.from_dict(tensors={"rm_scores": token_level_scores})
            output = self.ulysses_sharding_manager.postprocess_data(data=output)

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        self.reward_module._handle.reshard(True)

        output = output.to("cpu")
        torch.cuda.empty_cache()
        return output


class PotentialRewardModelWoker(Worker):
    """
    This class assigns partial reward for intermediate steps based on
    the potential of one partial solution leads to correct answer.
    """

    def __init__(self, config):
        super().__init__()
        import torch.distributed

        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
        self.config = config

        # build device mesh for Ulysses Sequence Parallel
        world_size = torch.distributed.get_world_size()
        from torch.distributed.device_mesh import init_device_mesh

        fsdp_size = self.config.model.fsdp_config.fsdp_size
        self.device_mesh = create_device_mesh(
            world_size=world_size, fsdp_size=fsdp_size
        )

        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.config.get(
            "ulysses_sequence_parallel_size", 1
        )
        dp = world_size // self.ulysses_sequence_parallel_size
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh(
                "cuda",
                mesh_shape=(dp, self.ulysses_sequence_parallel_size),
                mesh_dim_names=["dp", "sp"],
            )

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(
            self.ulysses_device_mesh
        )

        self.use_remove_padding = self.config.model.get("use_remove_padding", False)
        # self.config.micro_batch_size //= torch.distributed.get_world_size()

    def _build_tokenizer(self, config, local_path):
        # May require to switch tokenizer
        if self.config.model.input_tokenizer is None:
            self._do_switch_chat_template = False
        else:
            self._do_switch_chat_template = True
            input_tokneizer_local_path = copy_local_path_from_hdfs(
                self.config.model.input_tokenizer
            )

            self.input_tokenizer = hf_tokenizer(
                input_tokneizer_local_path,
                trust_remote_code=config.model.get("trust_remote_code", False),
            )
            self.tokenizer = hf_tokenizer(
                local_path,
                trust_remote_code=config.model.get("trust_remote_code", False),
            )
            try:
                self.tiktokenizer = AutoTikTokenizer.from_pretrained(local_path)
                print("Loaded TikTokenizer for faster encoding")
            except Exception as e:
                print(f"Failed to load TikTokenizer: {e}")
                self.tiktokenizer = None

        try:
            self.tiktokenizer = AutoTikTokenizer.from_pretrained(local_path)
            print("Loaded TikTokenizer for faster encoding")
        except Exception as e:
            print(f"Failed to load TikTokenizer: {e}")
        self.tiktokenizer = None

    def _build_model(self, config):
        # the following line is necessary
        from transformers import AutoModelForCausalLM, AutoConfig
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            ShardingStrategy,
            CPUOffload,
        )

        # download the checkpoint from hdfs
        local_path = copy_local_path_from_hdfs(config.model.path)
        self._build_tokenizer(config, local_path)

        # # NOTE (jiabao): we ignore the ulysis_sequence_parallel and rm_pad for simplicity here
        # note that we have to create model in fp32. Otherwise, the optimizer is in bf16, which is incorrect
        model_config = AutoConfig.from_pretrained(
            local_path,
            trust_remote_code=True,
        )
        init_context = get_init_weight_context_manager(
            use_meta_tensor=not model_config.tie_word_embeddings
        )

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reward_module = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=local_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                trust_remote_code=True,
            )
            reward_module.to(torch.bfloat16)
        auto_wrap_policy = get_fsdp_wrap_policy(
            module=reward_module, config=self.config.model.fsdp_config
        )

        fsdp_mesh = self.device_mesh
        sharding_strategy = get_sharding_strategy(fsdp_mesh)

        reward_module = FSDP(
            reward_module,
            param_init_fn=init_fn,
            use_orig_params=False,
            auto_wrap_policy=auto_wrap_policy,
            device_id=torch.cuda.current_device(),
            sharding_strategy=sharding_strategy,  # zero3
            sync_module_states=True,
            cpu_offload=CPUOffload(offload_params=True),
            forward_prefetch=False,
            device_mesh=self.device_mesh,
        )

        return reward_module

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get("external_lib", None))
        self.reward_module = self._build_model(config=self.config)
        torch.cuda.empty_cache()

    def detokenize_data(self, data: DataProto):
        src_tokenizer = self.input_tokenizer
        response_ids = data.batch["responses"]
        response_length = data.batch["attention_mask"][
            :, -response_ids.shape[-1] :
        ].sum(dim=-1)
        response_strs = src_tokenizer.batch_decode(
            response_ids, skip_special_tokens=True
        )
        detokenize_batch = {
            "response_length": response_length,
            "response_str": np.array(response_strs, dtype=object),
        }
        data.union(DataProto.from_single_dict(detokenize_batch))
        return data

    def split_by_steps(
        self, solution: str, num_step_per_chunk: int = 1
    ) -> List[Dict[str, str]]:
        split_token = "\n\n"
        steps = solution.split(split_token)

        split_results = []
        for i in range(0, len(steps) + 1, num_step_per_chunk):
            partial_solution = split_token.join(steps[:i])
            split_results.append(partial_solution)
        if len(steps) % num_step_per_chunk != 0:
            # Add the last step
            split_results.append(split_token.join(steps))
        if len(split_results) > self.config.max_num_chunk:
            # Reduce chunks
            inds = sorted(
                range(1, len(split_results) - 1),
                key=lambda x: len(split_results[x]) - len(split_results[x - 1]),
            )
            keep_inds = sorted(inds[-(self.config.max_num_chunk - 2) :])
            split_results = (
                split_results[:1]
                + [split_results[i] for i in keep_inds]
                + split_results[-1:]
            )
        return split_results

    def split_response(self, data: DataProto, index: int):
        self.tiktokenizer = None  # TODO (jiabao): debug tiktokenizer
        if self.config.split_method == "step":
            partial_solutions = self.split_by_steps(
                data.non_tensor_batch["response_str"][index],
                num_step_per_chunk=self.config.chunk_size,
            )
            if self.tiktokenizer is not None:
                ids = (
                    data.batch["responses"][
                        index, : data.batch["response_length"][index]
                    ]
                    .cpu()
                    .tolist()
                )
                prefix_to_pos = {
                    # self.tiktokenizer.decode(ids[:_i], allowed_special="all")
                    self.tiktokenizer.decode(ids[:_i])
                    .replace(self.tokenizer.pad_token, "")
                    .strip(): _i
                    for _i in range(len(ids) + 1)
                }
                prefix_to_pos[""] = 0
            else:
                special_tokens = self.tokenizer.all_special_tokens
                tokens = self.tokenizer.convert_ids_to_tokens(
                    data.batch["responses"][
                        index, : data.batch["response_length"][index]
                    ]
                )
                prefix_to_pos = {
                    self.tokenizer.convert_tokens_to_string(
                        [
                            tok
                            for tok in tokens[:_i]
                            if tok and tok not in special_tokens
                        ]
                    ).strip(): _i
                    for _i in range(len(tokens) + 1)
                }
            return [
                {"partial_solution": s, "position": prefix_to_pos[s.strip()] - 1}
                for s in partial_solutions
            ]

        elif self.config.split_method == "token":
            if data.batch["response_length"][index] / self.config.chunk_size > self.config.max_num_chunk:
                chunk_size = data.batch["response_length"][index] // self.config.max_num_chunk + 1
            else:
                chunk_size = self.config.chunk_size
            partial_solutions = [
                {
                    "partial_solution": self.tokenizer.decode(
                        data.batch["responses"][index, :i], skip_special_tokens=True
                    ),
                    "position": i - 1,
                }
                for i in range(
                    0, data.batch["response_length"][index] + 1, chunk_size
                )
            ]
            if data.batch["response_length"][index] % chunk_size != 0:
                partial_solutions.append(
                    {
                        "partial_solution": self.tokenizer.decode(
                            data.batch["responses"][
                                index, : data.batch["response_length"][index]
                            ],
                            skip_special_tokens=True,
                        ),
                        "position": data.batch["response_length"][index].item() - 1,
                    }
                )
            return partial_solutions

    def apply_choice_template(
        self, question: str, partial_sotluion: str, choices: List[str], max_tokens: int
    ) -> str:
        # TODO (jiabao): we may use a template file specified in config
        choice_template = [
            {
                "role": "system",
                # "content": "Based on the provided partial solution, answer the multiple-choice question with 4 possible answers (marked by A, B, C, D). Choose the best answer by writing its corresponding letter (either A, B, C, or D).",
                "content": "Based on the provided partial solution, answer the following multiple-choice question. Choose the best answer by writing its corresponding letter (e.g., A, B, C, ...).",
            },
            {
                "role": "user",
                "content": """Question:\n{question}\n\nPartial solution:\n{solution}\n\nChoices:\n{choices}""",
                # "content": """Question:\n{question}\n\nChoices:\n{choices}\n\nPartial solution:\n{solution}""",
            },
            {
                "role": "assistant",
                "content": """The answer is""",
                # "content": "Thinking" + " ".join(["..."] * 10) + " The answer is",
            },
        ]

        choice_str = "\n".join(
            [f"{chr(65 + i)}. {c}" for i, c in enumerate(choices)]
        )  # This is like A. B. C. D.

        if self.config.pad_to_same:
            cur_len = len(
                self.tokenizer(partial_sotluion, add_special_tokens=False)["input_ids"]
            )
            partial_sotluion += " ".join(["..."] * max(max_tokens - cur_len, 0))
        choice_template[1]["content"] = choice_template[1]["content"].format(
            question=question, solution=partial_sotluion, choices=choice_str
        )
        prompt = self.tokenizer.apply_chat_template(
            choice_template, continue_final_message=True, tokenize=False
        )
        return prompt

    def preprocess_rm(self, data: DataProto):
        # This function splits the response into steps and create a new reward batch
        # store the position of each split point,
        # store the mapping from reward batch idnex to origin data batch index

        rm_batch_dict = defaultdict(list)
        num_steps_per_response = []

        q_wrong_answers = defaultdict(set)
        if self.config.dynamic_distractor:
            # Unique wrong predicted answers for each question
            for i in range(data.batch.batch_size[0]):
                pred = data.non_tensor_batch["extracted_answers"][i]
                correctness = data.batch["answer_correctness"][i]
                if (
                    not correctness
                    and pred is not None
                    and pred != ""
                    and not pred.isalpha()
                ):
                    q_wrong_answers[data.non_tensor_batch["problem"][i]].add(pred)
            q_wrong_answers = {k: list(v) for k, v in q_wrong_answers.items()}

        # Each unique problem has the same choices
        q_choices = {}

        def mix_choices(gt_answer, other_choices):
            all_choices = other_choices + [gt_answer]
            random.shuffle(all_choices)
            gt_index = all_choices.index(gt_answer)
            return all_choices, gt_index

        for i in range(data.batch.batch_size[0]):
            question = data.non_tensor_batch["problem"][i]
            if question in q_choices:
                all_choices, gt_index = q_choices[question]
            else:
                distractors = q_wrong_answers.get(question, [])
                if len(distractors) < 3:
                    distractors += data.non_tensor_batch["distractors"][i].tolist()[
                        : 3 - len(distractors)
                    ]
                assert len(distractors) >= 3
                gt_answer = data.non_tensor_batch["reward_model"][i]["ground_truth"]
                all_choices, gt_index = mix_choices(gt_answer, distractors)
                q_choices[question] = (all_choices, gt_index)

            partial_solutions = self.split_response(data, i)
            max_tokens = len(
                self.tokenizer(
                    partial_solutions[-1]["partial_solution"], add_special_tokens=False
                )["input_ids"]
            )
            num_steps_per_response.append(len(partial_solutions))
            if (
                self.config.rule_terminal_potential
                and data.non_tensor_batch["extracted_answers"][i] != ""
            ) or self.config.constant_terminal_potential:
                # Cut final solution short to save a little compute
                partial_solutions[-1]["partial_solution"] = ""
            for step_cnt, partial_solution in enumerate(partial_solutions):
                prompt = self.apply_choice_template(
                    question=question,
                    partial_sotluion=partial_solution["partial_solution"],
                    choices=all_choices,
                    max_tokens=max_tokens,
                )
                rm_batch_dict["prompt"].append(prompt)
                rm_batch_dict["gt_index"].append(gt_index)
                rm_batch_dict["origin_index"].append(i)
                rm_batch_dict["num_choices"].append(len(all_choices))
                rm_batch_dict["step_pos"].append(partial_solution["position"])
                rm_batch_dict["step_cnt"].append(step_cnt)

        inputs = self.tokenizer(
            rm_batch_dict["prompt"],
            padding=True,
            return_tensors="pt",
            add_special_tokens=False,
        )

        rm_batch_dict = {
            k: torch.tensor(v)
            if isinstance(v[0], (int, float))
            else np.array(v, dtype=object)
            for k, v in rm_batch_dict.items()
        }

        rm_batch_dict["input_ids"] = inputs["input_ids"]
        rm_batch_dict["attention_mask"] = inputs["attention_mask"]

        return DataProto.from_single_dict(rm_batch_dict), num_steps_per_response

    def _compute_potential_reward(
        self, data: DataProto, rm_data: DataProto, gt_probs: torch.Tensor
    ):
        batch_size = data.batch.batch_size[0]
        token_level_scores = torch.zeros_like(
            data.batch["responses"], dtype=gt_probs.dtype
        )
        potentials = torch.zeros_like(token_level_scores)
        step_indicator = torch.zeros_like(data.batch["responses"])

        origin_indices = rm_data.batch["origin_index"].tolist()
        step_cnts = rm_data.batch["step_cnt"].tolist()
        step_positions = rm_data.batch["step_pos"]
        # Group the tuples (step_cnt, step_pos, prob) by origin_index
        grouped = defaultdict(list)
        for origin, cnt, pos, prob in zip(
            origin_indices, step_cnts, step_positions, gt_probs
        ):
            grouped[origin].append((cnt, pos, prob))

        response_prob_diff = {}
        # For each origin_index, sort the entries by step_cnt and concatenate step_pos and prob
        for origin, entries in grouped.items():
            sorted_entries = sorted(entries, key=lambda x: x[0])
            pos_list = [entry[1] for entry in sorted_entries]
            pos_list = pos_list[1:]  # Because the first is empty solution
            probs = torch.stack([entry[2] for entry in sorted_entries])
            probs *= self.config.potential_alpha
            token_level_scores[origin, pos_list] = probs[1:] - probs[:-1]
            step_indicator[origin, pos_list] = 1
            pos_list = [0] + pos_list
            for i, prob_i in enumerate(probs[:-1]):
                potentials[origin, pos_list[i] : pos_list[i + 1] + 1] = prob_i
            response_prob_diff[origin] = probs[-1] - probs[0]

        prob_diff = torch.stack([response_prob_diff[i] for i in range(batch_size)])
        avg_pr_per_step = prob_diff / step_indicator.sum(dim=-1)

        return token_level_scores, step_indicator, potentials, avg_pr_per_step

    def _forward_micro_batch(self, micro_batch: DataProto, data: DataProto):
        choice_indices = [
            self.tokenizer.encode(f" {chr(65 + i)}")[0]
            for i in range(micro_batch["num_choices"].max())
        ]
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            attention_mask = micro_batch["attention_mask"]
            max_length = attention_mask.sum(dim=-1).max()
            input_ids = input_ids[:, :max_length]
            attention_mask = attention_mask[:, :max_length]
            output = self.reward_module(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            rm_score = output.logits[
                torch.arange(input_ids.shape[0]), attention_mask.sum(dim=-1) - 1
            ]  # (batch_size, vocab_size)
            rm_score = rm_score[:, choice_indices]
            rm_score = torch.softmax(rm_score, dim=-1)
            indices = micro_batch["gt_index"].to(device=rm_score.device)
            rm_score = torch.gather(rm_score, dim=-1, index=indices.unsqueeze(-1))
            rm_score = rm_score.squeeze(-1)  # (batch_size,)

        # Mannually assign terminal probability
        if (
            self.config.rule_terminal_potential
            and data.non_tensor_batch["extracted_answers"][
                micro_batch["origin_index"][-1]
            ]
            != ""
        ):
            assert self.config.micro_batch_size == 1
            rm_score[-1] = data.batch["answer_correctness"][
                micro_batch["origin_index"][-1]
            ].item()
        elif self.config.constant_terminal_potential:
            assert self.config.micro_batch_size == 1
            rm_score[-1] = 1
        return rm_score.cpu()

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_rm_score(self, data: DataProto):
        # import pdb; pdb.set_trace()
        print(f"Rank {self.rank} computing reward")

        # Detokenize to string
        data = self.detokenize_data(data)
        # Split into steps and store the position of each split point
        rm_data, num_steps_per_response = self.preprocess_rm(data)
        if self.config.micro_batch_size > 1:
            # Group responses together to forward
            num_steps_per_response = [
                sum(num_steps_per_response[i : i + self.config.micro_batch_size])
                for i in range(
                    0, len(num_steps_per_response), self.config.micro_batch_size
                )
            ]

        # Compute potential reward for all job
        micro_batches = rm_data.batch.split(num_steps_per_response)
        assert (
            len(micro_batches)
            == data.batch.batch_size[0] // self.config.micro_batch_size
        )

        # TODO (jiabao): we may ues dynamic batch size to improve the efficiency
        gt_probs = []
        for batch_idx, micro_batch in enumerate(micro_batches):
            micro_batch_gt_probs = self._forward_micro_batch(micro_batch, data)
            gt_probs.append(micro_batch_gt_probs)
        gt_probs = torch.cat(gt_probs, dim=0)

        if self.config.save_probability:
            assert self.config.micro_batch_size == 1
            response_prob = gt_probs.split(num_steps_per_response)
            results = []
            for i in range(data.batch.batch_size[0]):
                results.append(
                    {
                        "uid": data.non_tensor_batch["uid"][i],
                        "probs": response_prob[i].tolist(),
                        "correctness": data.batch["answer_correctness"][i].item(),
                        "length": data.batch["response_length"][i].item(),
                        "extracted_answer": data.non_tensor_batch["extracted_answers"][
                            i
                        ],
                    }
                )
            import json
            json.dump(results, open(f"numina_1.5B_direct_qsc_{self.rank}.json", "w"))

        # map to potential reward and original dataset
        potential_rewards, step_indicator, potentials, avg_pr_per_step = (
            self._compute_potential_reward(data, rm_data, gt_probs)
        )

        output = DataProto.from_dict(
            tensors={
                "rm_scores": potential_rewards,
                "step_indicator": step_indicator,
                "potentials": potentials,
                "avg_pr_per_step": avg_pr_per_step,
            }
        )
        output = output.to("cpu")
        print(f"Rank {self.rank} finished RM")

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        if self.world_size > 1:
            self.reward_module._handle.reshard(True)

        torch.cuda.empty_cache()
        return output
