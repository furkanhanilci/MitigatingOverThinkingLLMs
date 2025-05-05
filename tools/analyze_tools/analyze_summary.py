"""
python tools/analyze_tools/analyze_summary.py --summary_path=logs/llm_sum/orig_qwen1_5B.json
"""
import json
import os
from typing import Dict, List, Union

import matplotlib.pyplot as plt
import numpy as np
import tqdm
from absl import app, flags
from jiwer import wer

flags.DEFINE_string(
    "summary_path",
    None,
    help="Path to the .json file that stores the summaries",
)
FLAGS = flags.FLAGS

def analyze_frequency(phase_info_list: List[Dict[str, List[Union[str, int]]]]):
    phase_to_freq = {}
    num_skipped = 0
    for example_idx, phase_info in enumerate(phase_info_list):
        phases = phase_info["phases"]
        start_indices = phase_info["start_indices"]
        end_indices = phase_info["end_indices"]

        should_skip = False
        for idx in range(len(phases)):
            curr_phase = phases[idx]
            curr_length = end_indices[idx] - start_indices[idx] + 1
            if curr_length <= 0:
                # skip invalid annotations
                should_skip = True
                break
        if should_skip:
            num_skipped += 1
            continue

        for idx in range(len(phases)):
            curr_phase = phases[idx]
            curr_length = end_indices[idx] - start_indices[idx] + 1
            if curr_phase not in phase_to_freq:
                phase_to_freq[curr_phase] = [curr_length]
            else:
                phase_to_freq[curr_phase].append(curr_length)
    print(f"{num_skipped} examples are skipped.")
    return phase_to_freq, num_skipped

def merge_freq_dict(phase_to_freq: dict):
    MAPPING = {
        "Solving the Equation": "Computing or Simplifying Expressions",
        "Solving the Quadratic Equation": "Computing or Simplifying Expressions",
        "Solving Equations": "Computing or Simplifying Expressions",
        "Solving Quadratic Equation": "Computing or Simplifying Expressions",
        "Comparing Coefficients": "Computing or Simplifying Expressions",
        "Assigning Coordinates": "Computing or Simplifying Expressions",
        "Combining Like Terms": "Computing or Simplifying Expressions",
    }
    copied_dict = phase_to_freq.copy()
    for k,v in MAPPING.items():
        if k not in copied_dict:
            continue
        else:
            poped_list = copied_dict.pop(k)
            copied_dict[v] = copied_dict[v] + poped_list
    return copied_dict

def visualize(phase_to_freq, labels: List[str], savename):
    sizes = [phase_to_freq[x] for x in labels]
    # Define custom colors
    colors = {
        "Understanding the Problem": "#1f77b4",  # Medium Blue
        "Reformulating the Problem": "#4a90d9",  # Lighter Blue

        "Applying Known Theorems/Properties": "#ff7f0e",  # Standard Orange
        "Computing or Simplifying Expressions": "#ffa64d",  # Lighter Orange
        "Substituting Known Values or Results": "#cc6600",  # Darker Orange

        "Reassess and Verify Local Steps": "#2ca02c",  # Standard Green
        "Reassess the Whole Solution": "#66cc66",  # Lighter Green

        "Breaking Down into Subproblems": "#d62728",  # Red
        "Exploring Alternative Approaches": "#9467bd",  # Purple
        "Finalize and Present the Answer": "#8c564b",  # Brown
    }
    color_list = [colors[label] for label in labels]

    fig, ax = plt.subplots(figsize=(10, 6))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=None, autopct='%1.1f%%', colors=color_list,
        startangle=140, wedgeprops={'edgecolor': 'white', 'linewidth': 1}, pctdistance=0.85
    )
    # Draw a white circle at the center to create the donut shape
    centre_circle = plt.Circle((0, 0), 0.5, fc='white')
    fig.gca().add_artist(centre_circle)

    # Add a legend
    ax.legend(
        wedges,
        labels,
        title="Problem-Solving Phases",
        loc="center left",
        bbox_to_anchor=(0.9, 0, 0.5, 1),
        columnspacing=1.5,
        # bbox_to_anchor=(1.1, 1),
    )

    # Display the plot
    plt.tight_layout()
    plt.savefig(savename)



def main(argv):
    run_logs =  json.load(open(FLAGS.summary_path))

    phase_info_list: List[Dict[str, str]] = []

    for log_dict in tqdm.tqdm(run_logs):
        question = log_dict["question"]
        answer = log_dict["answer"]
        llm_sum: str = log_dict["llm_sum"]

        llm_sum_lines = llm_sum.strip().split("\n")

        phases = []
        starts = []
        ends = []
        for idx, line in enumerate(llm_sum_lines):
            line = line.strip()
            if len(line) == 0:
                continue
            if line.startswith("[Phase"):
                end_position = line.index(":")
                extracted_phase = line[end_position+1:].strip()
                phases.append(extracted_phase)
            elif line.startswith("[Start]"):
                starts.append(line)
            elif line.startswith("[End]"):
                ends.append(line)
            else:
                print(line)
        if len(phases) == len(starts) == len(ends):
            pass
        else:
            import ipdb
            ipdb.set_trace()

        def fuzzy_matching(input_step, answer_steps: List[str],):
            match_scores = []
            best_matched_step_id = None
            for step_id, step in enumerate(answer_steps):
                if input_step in step or step in input_step:
                    best_matched_step_id = step_id
                    break
                else:
                    wer_score = wer(input_step, step)
                    match_score = 1 - wer_score
                    match_scores.append(match_score)
            if best_matched_step_id is None:
                best_matched_step_id = np.argmax(match_scores)
            return best_matched_step_id

        answer_steps = answer.split("\n\n")
        start_idices = []
        end_indices = []

        for step_idx in range(len(starts)):
            start_step = starts[step_idx]
            to_find = start_step.removeprefix("[Start]: ")
            tgt_id = fuzzy_matching(to_find, answer_steps,)
            start_idices.append(tgt_id)

            end_step: str = ends[step_idx]
            to_find = end_step.removeprefix("[End]: ")
            tgt_id = fuzzy_matching(to_find, answer_steps,)
            end_indices.append(tgt_id)

        phase_info_list.append(
            {
                "phases": phases,
                "start_indices": start_idices,
                "end_indices": end_indices,
            }
        )

    phase_to_freq, num_skipped = analyze_frequency(phase_info_list)
    phase_to_freq = merge_freq_dict(phase_to_freq)
    copied_phase_to_freq = {}
    total_steps = 0
    for k,v in phase_to_freq.items():
        print(k,np.sum(v))
        total_steps = total_steps + np.sum(v)
        copied_phase_to_freq[k] = np.sum(v)
    avg_step = total_steps / (len(run_logs) - num_skipped)
    print("average steps: ", avg_step)

    labels = [
        "Understanding the Problem", "Reformulating the Problem",
        "Applying Known Theorems/Properties", "Computing or Simplifying Expressions",
        "Substituting Known Values or Results",
        "Breaking Down into Subproblems", "Exploring Alternative Approaches",
        "Reassess and Verify Local Steps", "Reassess the Whole Solution",
        "Finalize and Present the Answer"
    ]
    "Understanding the Problem".removesuffix(".json")
    savename = f"figs/phase/{os.path.basename(FLAGS.summary_path).removesuffix('.json')}.pdf"
    visualize(copied_phase_to_freq, labels, savename)

if __name__ == "__main__":
    app.run(main)


