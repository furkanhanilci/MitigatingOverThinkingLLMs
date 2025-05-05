"""
python tools/analyze_tools/llm_summary.py --data_path_prefix=logs/math500/orig_qwen1_5B
"""

import json
import os
from typing import Dict, List

from absl import app, flags
from openai import AzureOpenAI
from tqdm.auto import tqdm as auto_tqdm

import tools.prompts as prompts

flags.DEFINE_string(
    "data_path_prefix",
    None,
    help="Path to the .json file that stores the data and model outputs",
)

FLAGS = flags.FLAGS


def read_file(data_path_prefix: str, rank: int):
    file_path = os.path.join(data_path_prefix, f"rank{rank}.jsonl")
    all_logs = []
    if not os.path.exists(file_path):
        return all_logs
    with open(file_path) as f:
        for line in f:
            content = json.loads(line)
            all_logs.append(content)
    return all_logs

def query_api(**kwargs):
    client = AzureOpenAI(
        api_version="2024-08-01-preview",
        azure_endpoint="",
        api_key="",
    )
    result = client.chat.completions.create(
        model="gpt-4o",
        **kwargs,
    )
    return result

def main(argv):
    data_path_prefix = FLAGS.data_path_prefix
    all_logs = []
    for rank in [0,1,2,3,4,5,6,7]:
        all_logs += read_file(data_path_prefix, rank)


    all_sum_logs = []
    num_queries = len(all_logs)
    prog_bar = auto_tqdm(range(num_queries))
    for data_idx in range(num_queries):
        curr_log = all_logs[data_idx]
        model_answers = curr_log["model_answer"]
        problem = curr_log["question"]
        for ans in model_answers:
            if "</think>" not in ans:
                continue
            curr_answer = ans.split("</think>", 1)[0]
            break

        qa_input = f"Problem:\n{problem}\n\nSolution:\n{curr_answer}"
        user_prompt = prompts.SUMMARIZE_PROMPT + "\n\n" + qa_input.strip()
        message = [{"role": "user", "content": user_prompt}]

        try:
            response = query_api(
                messages=message,
                temperature=0.0, max_tokens=10000,n=1,
            )
            summarized_result = response.choices[0].message.content.strip()
        except:
            prog_bar.update(1)
            continue

        analyze_log = {
            "question": problem,
            "answer": curr_answer,
            "llm_sum": summarized_result
        }

        all_sum_logs.append(analyze_log)
        prog_bar.update(1)

    basename = os.path.basename(data_path_prefix)
    save_name = os.path.join("logs/llm_sum/", basename + ".json")
    if not os.path.exists(os.path.dirname(save_name)):
        os.makedirs(os.path.dirname(save_name))
    with open(save_name, "w") as f:
        json.dump(all_sum_logs, f, indent=4)


if __name__ == "__main__":
    app.run(main)
