"""
Convert the historical AIME/AMC data to the target format (Parquet) required by Verl.
```
python examples/data_preprocess/preprocess_past_aime_amc.py \
    --dataset_cache_path=aux_data \
    --model_family=qwen \
    --max_length=4000 \
    --save_dir=data/past_aime_amc_qwq/length4000
```
"""
import copy
import os
import datasets
from absl import app, flags
from typing import List, Dict

from verl.utils.hdfs_io import makedirs

flags.DEFINE_string(
    "dataset_cache_path",
    None,
    help="The directory that saves all processed subsets",
)
flags.DEFINE_string(
    "save_dir",
    None,
    help="The directory to save the processed data.",
)
flags.DEFINE_string(
    "max_length",
    None,
    required=False,
    help="The directory to save the processed data.",
)
flags.DEFINE_enum(
    "model_family",
    "deepseek",
    enum_values=["deepseek", "qwen"],
    required=False,
    help="The model family of the LLM (i.e., which company train this model).",
)
FLAGS = flags.FLAGS

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, "
    "and the Assistant solves it."
    " The assistant first thinks about the reasoning process in the mind and then provides the user"
    " with the answer. The reasoning process and answer are enclosed within <think></think> and"
    " <answer></answer> tags, respectively, i.e., <think> reasoning process here</think>"
    "<answer> answer here</answer>."
)

def main(argv):
    # everything has been processed, with two extra keyss problem and distractors
    train_dataset = datasets.load_dataset("PRIME-RL/Eurus-2-RL-Data", split="train")
    def filter_fn(example):
        if example["data_source"] not in ["numina_amc_aime"]:
            return False
        return True
    train_dataset = train_dataset.filter(filter_fn, batched=False)
    level_info = [6 for _ in range(len(train_dataset))]
    print("number of columns: ", len(level_info))
    train_dataset = train_dataset.add_column("level", level_info)

    test_dataset = datasets.load_dataset(
        "parquet",
        data_files=[
            os.path.join(FLAGS.dataset_cache_path, "test_aime.parquet"),
        ]
    )["train"]
    test_dataset = test_dataset.remove_columns("problem")
    assert set(test_dataset.column_names) == set(train_dataset.column_names)


    system_prompt_w_length = SYSTEM_PROMPT[:]
    if FLAGS.max_length is not None:
        max_length = int(FLAGS.max_length)
        system_prompt_w_length = system_prompt_w_length + (
            f" The output of the assistant should be within {max_length} tokens."
        )
    print(system_prompt_w_length)

    def get_process_fn(split: str, model_family: str):
        def process_fn(example, idx):
            if model_family == "deepseek":
                instruction = "Let's think step by step and output the final answer within \\boxed{}."
            elif model_family == "qwen":
                instruction = "Please reason step by step, and put your final answer within \\boxed{}."
            else:
                raise NotImplementedError()
            prompt: List[Dict[str, str]] = example.pop('prompt')
            if split == "train":
                assert len(prompt) == 2 and prompt[1]["role"] == "user"
                orig_question = prompt[1]["content"].removesuffix("\n\nPresent the answer in LaTex format: \\boxed{Your answer}")
                question = orig_question + " " + instruction
            else:
                assert len(prompt) == 1 and prompt[0]["role"] == "user"
                orig_question = prompt[0]["content"]
                question = orig_question + " " + instruction

            new_prompt = [
                {"role": "system", "content": system_prompt_w_length},
                {"role": "user", "content": question}
            ]
            data = copy.deepcopy(example)
            data["prompt"] = new_prompt
            return data
        return process_fn

    keep_columns = {"data_source", "prompt", "ability", "reward_model", "extra_info", "level"}
    train_dataset = train_dataset.map(
        function=get_process_fn("train", model_family=FLAGS.model_family),
        with_indices=True,
        remove_columns=list(set(train_dataset.features) - keep_columns),
    )
    test_dataset = test_dataset.map(
        function=get_process_fn("test", model_family=FLAGS.model_family),
        with_indices=True,
        remove_columns=list(set(test_dataset.features) - keep_columns),
    )

    print("example training example:")
    print(train_dataset[0]["prompt"])
    print()

    print("example testing example:")
    print(test_dataset[0]["prompt"])

    save_dir = FLAGS.save_dir
    makedirs(save_dir)

    assert "level" in train_dataset.column_names
    assert "level" in test_dataset.column_names

    train_dataset.to_parquet(os.path.join(save_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(save_dir, "test.parquet"))

if __name__ == "__main__":
    app.run(main)

