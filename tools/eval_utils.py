import os
import json
from typing import Dict, List

import datasets
import numpy as np

DATASET_TO_SAMPLING_PARAMS = {
    "math500": dict(
        temperature=0.6,
        top_p=0.95,
        n=16,
        max_tokens=32_000,
    ),
    "aime": dict(
        temperature=0.6,
        top_p=0.95,
        n=64,
        max_tokens=32_000,
    ),
    "aime2223": dict(
        temperature=0.6,
        top_p=0.95,
        n=16,
        max_tokens=32_000,
    ),
    "amc23": dict(
        temperature=0.6,
        top_p=0.95,
        n=64,
        max_tokens=32_000,
    ),
    "olympiad_bench": dict(
        temperature=0.6,
        top_p=0.95,
        n=16,
        max_tokens=32_000,
    ),
}

def load_local_amc23(rank: int, world_size: int):
    data_path = "local_ds/AMC23_test.jsonl"
    dataset = []
    with open(data_path, "r") as f:
        for line in f:
            data = json.loads(line)
            data["solution"] = data["answer"]
            dataset.append(data)
    num_examples = len(dataset)
    size_per_rank = num_examples // world_size
    if rank < world_size - 1:
        local_ds = dataset[rank * size_per_rank: (rank+1) * size_per_rank]
    else:
        local_ds = dataset[rank * size_per_rank:]
    print(f"{len(local_ds)} examples loaded.")
    return local_ds

def load_local_olympiad_bench(rank: int, world_size: int):
    data_path = "local_ds/olympiad_bench.jsonl"
    dataset = []
    with open(data_path, "r") as f:
        for line in f:
            data = json.loads(line)
            data["answer"] = data["final_answer"]
            dataset.append(data)
    num_examples = len(dataset)
    size_per_rank = num_examples // world_size
    if rank < world_size - 1:
        local_ds = dataset[rank * size_per_rank: (rank+1) * size_per_rank]
    else:
        local_ds = dataset[rank * size_per_rank:]
    print(f"{len(local_ds)} examples loaded.")
    return local_ds

def load_math500(rank: int, world_size: int):
    hf_data = datasets.load_dataset("HuggingFaceH4/MATH-500", split="test")
    dataset = []
    for data in hf_data:
        converted_data = {
            "question": data["problem"],
            "solution": data["solution"],
            "answer": data["answer"],
        }
        dataset.append(converted_data)
    num_examples = len(dataset)
    size_per_rank = num_examples // world_size
    if rank < world_size - 1:
        local_ds = dataset[rank * size_per_rank: (rank+1) * size_per_rank]
    else:
        local_ds = dataset[rank * size_per_rank:]
    print(f"{len(local_ds)} examples loaded.")
    return local_ds

def load_aime2024(rank: int, world_size: int):
    data_path = "local_ds/aime24/test.jsonl"
    dataset = []
    with open(data_path, "r") as f:
        for line in f:
            data = json.loads(line)
            data["solution"] = data["answer"]
            dataset.append(data)
    num_examples = len(dataset)
    if world_size == 8:
        size_per_rank = 4
    elif world_size == 4:
        size_per_rank = 8
    else:
        size_per_rank = num_examples // world_size
    if rank < world_size - 1:
        local_ds = dataset[rank * size_per_rank: (rank+1) * size_per_rank]
    else:
        local_ds = dataset[rank * size_per_rank:]
    print(f"{len(local_ds)} examples loaded.")
    return local_ds

def load_aime2223(rank: int, world_size: int):
    data_path = "local_ds/aime2223.jsonl"
    if os.path.exists(data_path):
        dataset = []
        with open(data_path) as f:
            for line in f:
                dataset.append(json.loads(line))
    else:
        current_file = os.path.abspath(__file__)
        verl_root = os.path.dirname(os.path.dirname(current_file))
        raw_data_path = os.path.join(verl_root, 'aux_data/test_aime.parquet')

        hf_dataset = datasets.load_dataset("parquet", data_files=raw_data_path)["train"]

        dataset = []
        for item in hf_dataset:
            extracted_question = item["problem"]
            extracted_solution = item["reward_model"]["ground_truth"]
            data_source = item["data_source"]
            if data_source not in ["AIME22", "AIME23"]:
                continue
            converted_data = {
                "question": extracted_question,
                "solution": extracted_solution,
                "answer": extracted_solution,
            }
            dataset.append(converted_data)

        with open(data_path, "w") as f:
            for item in dataset:
                f.write(json.dumps(item) + "\n")

    num_examples = len(dataset)
    size_per_rank = num_examples // world_size
    if rank < world_size - 1:
        local_ds = dataset[rank * size_per_rank: (rank+1) * size_per_rank]
    else:
        local_ds = dataset[rank * size_per_rank:]
    print(f"{len(local_ds)} examples loaded.")
    return local_ds


