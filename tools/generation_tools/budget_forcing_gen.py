"""
Implement the budget-forcing generation of s1 by:
    1. Removing the dependency of lm-eval-harness.
    2. Support sampling multiple responses from one question.
Reference:
    https://github.com/simplescaling/s1/blob/main/eval/lm-evaluation-harness/lm_eval/models/vllm_causallms.py#L219

python tools/s1_generation.py --rank=0 --world_size=1 --batch_size=2 \
    --model_name=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --orig_model_name=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --save_name=test \
    --max_tokens_thinking=1000 \
    --dataset_name=math500
"""
import copy
import json
import os
from typing import List, Optional, Union

import torch
from absl import app, flags
from tqdm.auto import tqdm as auto_tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from tools.eval_utils import (
    DATASET_TO_SAMPLING_PARAMS,
    load_aime2024,
    load_local_amc23,
    load_local_olympiad_bench,
    load_math500,
)

flags.DEFINE_integer("rank", None, help="The rank for current progress.")
flags.DEFINE_integer("world_size", None, help="Total number of GPUs available.")
flags.DEFINE_integer("batch_size", 4, help="The generation batch size.")
flags.DEFINE_integer(
    "max_tokens_thinking",
    None,
    required=True,
    help="The thinking token budget.",
)
flags.DEFINE_string(
    "orig_model_name",
    None,
    required=True,
    help="The original Huggingface name of evaluation model before training."
)
flags.DEFINE_string(
    "model_name",
    None,
    required=True,
    help="The evaluation model."
)
flags.DEFINE_string(
    "save_name",
    None,
    required=True,
    help="where to save the generation results."
)
flags.DEFINE_enum(
    "dataset_name",
    None,
    required=True,
    enum_values=["math500", "aime", "amc23", "olympiad_bench"],
    help="The evaluation dataset."
)
flags.DEFINE_integer(
    "tp_size",
    1,
    help="tensor parallel size of vLLM."
)

FLAGS = flags.FLAGS

MODEL_NANE_TO_SPECIAL_TOKENS = {
    "simplescaling/s1": {
        "thinking_start": "<|im_start|>think",
        "thinking_end": "<|im_start|>answer",
        "thinking_end_max": "<|im_start|>answer\nFinal Answer:",
        "until_thinking": "<|im_start|>",
    },
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B":{
        "thinking_start": "<｜Assistant｜><think>\n",
        "thinking_end": "</think>\n\n",
        "thinking_end_max": "</think>\n\n**Final Answer:**\n\n",
        "until_thinking": "</think>",
    },
    "agentica-org/DeepScaleR-1.5B-Preview":{
        "thinking_start": "<｜Assistant｜><think>\n",
        "thinking_end": "</think>\n\n",
        "thinking_end_max": "</think>\n\n**Final Answer:**\n\n",
        "until_thinking": "</think>",
    },
    "Qwen/QwQ-32B":{
        "thinking_start": "<|im_start|>assistant\n<think>",
        "thinking_end": "</think>\n\n",
        "thinking_end_max": "</think>\n\nThe final answer is",
        "until_thinking": "</think>",
    },

}

class S1StyleVLLM():
    def __init__(self, model_name):
        max_model_len = 32_768
        self.model = LLM(
            model_name,
            tensor_parallel_size=FLAGS.tp_size,
            dtype=torch.bfloat16,
            max_model_len=max_model_len,
            disable_custom_all_reduce=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.special_token_map = MODEL_NANE_TO_SPECIAL_TOKENS[FLAGS.orig_model_name]

    @staticmethod
    def modify_gen_kwargs(kwargs: dict) -> dict:
        # sampling_params
        do_sample = kwargs.pop("do_sample", None)
        if do_sample is False and "temperature" not in kwargs:
            kwargs["temperature"] = 0.0
        # hf defaults
        kwargs["skip_special_tokens"] = kwargs.get("skip_special_tokens", False)
        kwargs["spaces_between_special_tokens"] = kwargs.get(
            "spaces_between_special_tokens", False
        )
        return kwargs

    def tok_encode(
        self,
        string: Union[str, List[str]],
        left_truncate_len: int = None,
        truncation: bool = False,
    ) -> Union[List[int], List[List[int]]]:
        encoding: Union[List[List[int]], List[int]] = self.tokenizer(
            string,
            add_special_tokens=False,
            truncation=truncation,
            return_attention_mask=False,
        ).input_ids

        # left-truncate the encoded context to be at most `left_truncate_len` tokens long
        if left_truncate_len:
            if not isinstance(string, str):
                encoding = [enc[-left_truncate_len:] for enc in encoding]
            else:
                encoding = encoding[-left_truncate_len:]

        return encoding

    def _model_generate(
        self,
        requests: List[List[int]] = None,
        max_tokens: int = None,
        stop: Optional[List[str]] = None,
        **kwargs,
    ):
        kwargs = self.modify_gen_kwargs(kwargs)

        outputs_thinking = None
        # import ipdb
        # ipdb.set_trace()
        sample_n = kwargs.pop("n")
        if any(["thinking" in k for k in kwargs]):
            print("Separating thinking and answering generation.")
            thinking_start = self.special_token_map["thinking_start"]
            thinking_end = self.special_token_map["thinking_end"]
            until_thinking = [self.special_token_map["until_thinking"]]
            if "until_thinking_2" in kwargs:
                until_thinking.append(kwargs.pop("until_thinking_2"))
            if stop is not None:
                until_thinking.extend(stop)
            print(f"Thinking start: {thinking_start}, Thinking end: {thinking_end}, Stop: {until_thinking}")
            thinking_start_tok = self.tok_encode(thinking_start)
            thinking_end_tok = self.tok_encode(thinking_end)
            # thinking_end_max = thinking_end + "\nFinal Answer:"
            thinking_end_max = self.special_token_map["thinking_end_max"]
            thinking_end_max_tok = self.tok_encode(thinking_end_max)
            newline_tok = self.tok_encode("\n")
            # Cast to list to avoid `dictionary changed size during iteration`
            sampling_params_thinking = {k.replace("_thinking", ""): kwargs.pop(k) for k, v in list(kwargs.items()) if "thinking" in k}
            # Add all other kwargs but keep sampling_params_thinking version if duplicate key
            sampling_params_thinking = {**kwargs, **sampling_params_thinking}
            if "max_tokens" in sampling_params_thinking:
                if sampling_params_thinking["max_tokens"] == "auto":
                    # Leave 100 tokens for answer
                    sampling_params_thinking["max_tokens"] = max_tokens - max([len(x) for x in requests]) - len(thinking_start_tok) - len(thinking_end_max_tok) - 100
                    print(f"Auto setting max_tokens_thinking to {sampling_params_thinking['max_tokens']}")
                else:
                    sampling_params_thinking["max_tokens"] = int(sampling_params_thinking["max_tokens"])
            else:
                sampling_params_thinking["max_tokens"] = max_tokens
            until_thinking_tok = self.tok_encode(until_thinking)
            sampling_params_thinking["stop"] = until_thinking

            # Bairu: insert `n` into the sampling parameter here.
            sampling_params_thinking["n"] = sample_n

            requests = [req + thinking_start_tok for req in requests]

            sampling_params = SamplingParams(**sampling_params_thinking)
            vllm_inputs = [{"prompt_token_ids": requests[i]} for i in range(len(requests))]
            outputs_thinking = self.model.generate(
                # prompt_token_ids=requests,
                prompts=vllm_inputs,
                sampling_params=sampling_params,
                # use_tqdm=True if self.batch_size == "auto" else False,
                use_tqdm=False,
            )

            requests_after_multi_sampling_for_each_query = []
            for i, o in enumerate(outputs_thinking):
                # Bairu: delete this line. We sample multiple outputs for each question
                # assert len(o.outputs) == 1

                # Bairu: for-loop to change each sampled response
                for sample_id in range(len(o.outputs)):
                    cont = list(o.outputs[sample_id].token_ids)
                    # When using `stop`, the stop text will not be in the text, but still in the token_ids so remove it
                    for toks in until_thinking_tok:
                        if cont[-len(toks):] == toks:
                            cont = cont[:-len(toks)]

                    # TODO: (bairu) refactor the if-else logic here
                    if o.outputs[sample_id].finish_reason == "length":
                        # \n appears a lot so a decent chance it happend to just be the last token in which case we don't need to add a newline
                        if (o.outputs[sample_id].text[-1] == "\n") or (thinking_start[0] == "\n"):
                            # Bairu: request[i]: i-th question; cont: j-th sampled response
                            curr_request = requests[i] + cont + thinking_end_max_tok
                            requests_after_multi_sampling_for_each_query.append(curr_request)
                            outputs_thinking[i].outputs[sample_id].text = thinking_start + outputs_thinking[i].outputs[sample_id].text + thinking_end_max
                        else:
                            # Bairu: request[i]: i-th question; cont: j-th sampled response
                            curr_request = requests[i] + cont + newline_tok + thinking_end_max_tok
                            requests_after_multi_sampling_for_each_query.append(curr_request)
                            # requests[i] += cont + newline_tok + thinking_end_max_tok
                            outputs_thinking[i].outputs[sample_id].text = thinking_start + outputs_thinking[i].outputs[sample_id].text + "\n" + thinking_end_max
                    else:
                        curr_request = requests[i] + cont + thinking_end_tok
                        requests_after_multi_sampling_for_each_query.append(curr_request)
                        # requests[i] += cont + thinking_end_tok
                        outputs_thinking[i].outputs[sample_id].text = thinking_start + outputs_thinking[i].outputs[sample_id].text + thinking_end

        # we must set n=1 here.
        sampling_params = SamplingParams(max_tokens=max_tokens, stop=stop, n=1, **kwargs)
        vllm_inputs = [
            {"prompt_token_ids": requests_after_multi_sampling_for_each_query[i]}
            for i in range(len(requests_after_multi_sampling_for_each_query))
        ]
        outputs = self.model.generate(
            # prompt_token_ids=requests_after_multi_sampling_for_each_query,
            prompts=vllm_inputs,
            sampling_params=sampling_params,
            # use_tqdm=True if self.batch_size == "auto" else False,
            use_tqdm=False,
        )
        all_grouped_outputs = []
        for group_id in range(len(requests)):
            curr_group_outputs = []
            for local_idx in range(sample_n):
                global_idx = group_id * sample_n + local_idx
                outputs[global_idx].outputs[0].text = outputs_thinking[group_id].outputs[local_idx].text + outputs[global_idx].outputs[0].text
                outputs[global_idx].outputs[0].token_ids = outputs_thinking[group_id].outputs[local_idx].token_ids + outputs[global_idx].outputs[0].token_ids
                curr_group_outputs.append(outputs[global_idx])

            all_grouped_outputs.append(curr_group_outputs)
        return all_grouped_outputs

        # if outputs_thinking is not None:
        #     for i, o in enumerate(outputs):
        #         assert len(o.outputs) == 1
        #         outputs[i].outputs[0].text = outputs_thinking[i].outputs[0].text + outputs[i].outputs[0].text
        # return outputs



def main(argv):
    rank = FLAGS.rank
    world_size = FLAGS.world_size
    batch_size = FLAGS.batch_size
    dataset_name = FLAGS.dataset_name

    if dataset_name == "math500":
        local_ds = load_math500(rank, world_size)
    elif dataset_name == "aime":
        local_ds = load_aime2024(rank, world_size)
    elif dataset_name == "amc23":
        local_ds = load_local_amc23(rank, world_size)
    elif dataset_name == "olympiad_bench":
        local_ds = load_local_olympiad_bench(rank, world_size)
    else:
        raise ValueError()

    # model_name = "simplescaling/s1"
    model_name = FLAGS.model_name
    llm_model = S1StyleVLLM(model_name)
    tokenizer = llm_model.tokenizer

    sampling_params_dict = DATASET_TO_SAMPLING_PARAMS[dataset_name]
    if "qwq" in model_name.lower():
        sampling_params_dict["n"] = 8
        print("Only sample 8 responses for QwQ due to the large size")

    max_tokens_thinking = FLAGS.max_tokens_thinking
    # max_tokens = 1000
    max_tokens = 32000

    local_ds = local_ds
    output_logs = []
    prog_bar = auto_tqdm(range(len(local_ds)))


    batched_prompts = []
    for data_idx in range(len(local_ds)):
        question = local_ds[data_idx]["question"]
        if FLAGS.model_name in ["simplescaling/s1"]:
            message = [
                {
                    "role": "system",
                    "content": (
                        "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
                    ),
                },
                {"role": "user", "content": question},
            ]
        else:
            instruction = "Let's think step by step and output the final answer within \\boxed{}."
            message = [{"role": "user", "content": question + " " + instruction}]
        prompt = tokenizer.apply_chat_template(
            message,
            add_generation_prompt=False,
            tokenize=False,
        )
        prompt_tokens = tokenizer(
            prompt, add_special_tokens=False,
        ).input_ids
        batched_prompts.append(prompt_tokens)
        if len(batched_prompts) == batch_size or data_idx == len(local_ds) - 1:
            grouped_outputs = llm_model._model_generate(
                batched_prompts,
                max_tokens_thinking=max_tokens_thinking,
                max_tokens=max_tokens,
                temperature=sampling_params_dict["temperature"],
                n=sampling_params_dict["n"],
                do_sample=True,
                top_p=sampling_params_dict["top_p"],
            )
            answers = [
                [
                    output.outputs[0].text for output in output_group
                ]
                for output_group in grouped_outputs
            ]
            answer_token_ids_list = [
                [
                    output.outputs[0].token_ids for output in output_group
                ]
                for output_group in grouped_outputs
            ]
            for idx in range(len(answers)):
                curr_data_idx = data_idx - len(answers) + idx + 1
                exec_log = copy.deepcopy(local_ds[curr_data_idx])
                # exec_log.pop("input")
                exec_log["model_answer"] = answers[idx]
                answer_tokens_ids = answer_token_ids_list[idx]
                exec_log["model_answer_length"] = [len(x) for x in answer_tokens_ids]

                output_logs.append(exec_log)

            batched_prompts.clear()

        prog_bar.update(1)

    # save_path = f"logs/{dataset_name}_{FLAGS.model_name}_rank{rank}.jsonl"
    if str(max_tokens_thinking) not in FLAGS.save_name:
        save_path = f"logs/s1/{dataset_name}/{FLAGS.save_name}_budget{max_tokens_thinking}_maxtoken{max_tokens}/rank{rank}.jsonl"
    else:
        save_path = f"logs/s1/{dataset_name}/{FLAGS.save_name}_maxtoken{max_tokens}/rank{rank}.jsonl"
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    with open(save_path, "w") as f:
        for item in output_logs:
            f.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    app.run(main)



