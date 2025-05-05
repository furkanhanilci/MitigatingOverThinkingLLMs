"""
python tools/eval_accuracy.py --data_path_prefix=logs/math500_qwen1.5B
"""
import json
import os
from threading import Lock

import numpy as np
import tqdm
from absl import app, flags

from verl.utils.reward_score.qwen_math import compute_score

flags.DEFINE_string(
    "data_path_prefix",
    None,
    help="Path to the .json file that stores the data and model outputs",
)
flags.DEFINE_boolean(
    "do_analyze",
    False,
    help="Whether analyze the correct/error statistics."
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

def main(argv):
    data_path_prefix = FLAGS.data_path_prefix
    all_logs = []
    for rank in [0,1,2,3,4,5,6,7]:
    # for rank in [0]:
        all_logs += read_file(data_path_prefix, rank)

    correct = 0.0
    all_scores = []
    all_lengths = []
    all_correct_lengths = []
    all_wrong_lengths = []
    for example in tqdm.tqdm(all_logs):
        gt_solution = example["answer"]
        if "olympia" in data_path_prefix:
            gt_solution = gt_solution[0]
        model_answers = example["model_answer"]
        model_answer_lengths = example["model_answer_length"]

        model_scores = []
        model_numbers = []
        results_cache = {}
        lock = Lock()
        for ans_id, x in enumerate(model_answers):
            ans_len = model_answer_lengths[ans_id]
            score, ext_number = compute_score(x, gt_solution, results_cache, lock)
            model_scores.append(score)
            all_lengths.append(ans_len)
            if score == 1.0:
                all_correct_lengths.append(ans_len)
            else:
                all_wrong_lengths.append(ans_len)
            model_numbers.append(ext_number)

        curr_score = np.mean(model_scores)
        correct += curr_score
        all_scores += model_scores
        print(model_numbers, gt_solution)

    pass_1 = correct / len(all_logs)

    print("avg pass@1: ", pass_1)

    avg_length = np.mean(all_lengths)
    print(f"average length: {avg_length}")
    print(len(all_logs))
    print(len(all_lengths))
    highlighted_percentiles = [20, 40, 50, 60, 80, 90, 99]
    highlighted_values = np.percentile(all_correct_lengths, q=highlighted_percentiles)
    print("correct length distribution:", highlighted_values)
    highlighted_percentiles = [20, 40, 50, 60, 80, 90, 99]
    highlighted_values = np.percentile(all_wrong_lengths, q=highlighted_percentiles)
    print("wrong length distribution:", highlighted_values)



if __name__ == "__main__":
    app.run(main)
