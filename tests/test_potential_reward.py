# This file contains functions to test verl.workeres.fsdp_workers.PotentialRewardModelWoker
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer

from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
from verl import DataProto
from verl.workers.fsdp_workers import PotentialRewardModelWoker


model_name = "Qwen/Qwen2.5-Math-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)


def split2steps(solution: str) -> List[Dict[str, str]]:
    split_token = "\n\n"
    steps = solution.split(split_token)

    split_results = []
    for i in range(1, len(steps)):
        partial_solution = split_token.join(steps[:i])
        split_results.append(partial_solution)
    return split_results


def test_split2step():
    test_string = "\n\n".join(["hi how are you? " * i for i in range(1, 5)])
    partials = split2steps(test_string)
    print(partials)
    print("===" * 100)

    full_ids = tokenizer(test_string)["input_ids"]
    print(full_ids)
    print("---" * 100)
    for part in partials:
        ids = tokenizer(part)["input_ids"]
        pos = len(ids)
        if full_ids[:pos] != ids:
            print("not correct")
            print(full_ids[:pos])
            print(ids)
            print(partials)
            print(part)
        # assert , "Not correct"


def build_dataloader():
    dataset = RLHFDataset(
        parquet_files="./data_cache/debug_math/train.parquet",
        tokenizer=tokenizer,
        prompt_key="prompt",
        max_prompt_length=512,
        filter_prompts=True,
        return_raw_chat=True,
        truncation="error",
    )

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_fn,
    )

    return dataloader


def main():
    dataloader = build_dataloader()

    for batch in dataloader:
        batch = DataProto.from_dict(batch)

        import ipdb

        ipdb.set_trace()
    # data = worker.get_data()k
    # score = worker.compute_rm_score(data)
    # print(score)


if __name__ == "__main__":
    main()
