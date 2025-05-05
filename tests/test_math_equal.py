import os

import json
from tqdm import tqdm
from verl.evaluation import REWARD_REGISTRY, pass_at_k
from datasets import load_from_disk
import time


def get_all_pass_k(all_results):
    pass_at_ks = {}
    for k in [1, 2, 4, 5, 8, 10]:
        res = pass_at_k(all_results, k)
        if res is not None:
            pass_at_ks[f"pass@{k}"] = res
    return pass_at_ks


def main():
    # respath = "/Users/jiyi/Documents/2025-Projects/yj-verl/evalres/trialrun/model@Qwen2.5-Math-7B,step@None,prompt@qwen_jxhe,time@01-30-19-36-20/results_data"
    # respath = "evalres/trialrun/model@Qwen2.5-Math-7B,step@None,prompt@qwen_jxhe,time@01-30-19-36-20/results_data"
    # respath = "evalres/trialrun/model@Qwen2.5-Math-7B,step@None,prompt@qwen_jxhe,time@01-30-19-36-20/results_data"
    respath = "evalres/trialrun/model@Qwen2.5-Math-1.5B,step@None,prompt@qwen_jxhe,time@01-31-14-08-03/results_data"
    # respath = ""
    dataset = load_from_disk(respath)

    qwen_rm = REWARD_REGISTRY.get("qwen_math")
    deepseek_rm = REWARD_REGISTRY.get("deepseek_math")

    inconsistent = 0
    qwen_results = []
    deepseek_results = []
    qwen_time = 0.0
    deepseek_time = 0.0
    for sample in tqdm(dataset):
        now = time.time()
        qwen_res = qwen_rm(sample["response"], sample["gt_answer"])
        qwen_time += time.time() - now
        now = time.time()
        deepseek_res = deepseek_rm(sample["response"], sample["gt_answer"])
        deepseek_time += time.time() - now
        if qwen_res != deepseek_res:
            print(f"Qwen: {qwen_res}, DeepSeek: {deepseek_res}")
            print(f"GT: {sample['gt_answer']}, Res: {sample['response']}")
            print("===" * 20)
            inconsistent += 1
        qwen_results.append(
            {
                "uid": sample["uid"],
                "match": float(qwen_res),
            }
        )
        deepseek_results.append(
            {
                "uid": sample["uid"],
                "match": float(deepseek_res),
            }
        )

    qwenpassk = get_all_pass_k(qwen_results)
    deepseekpassk = get_all_pass_k(deepseek_results)

    print(json.dumps(qwenpassk, indent=2))
    print("===" * 20)
    print(json.dumps(deepseekpassk, indent=2))

    print("In all ", inconsistent, "inconsistent results")
    print("Qwen time", qwen_time)
    print("DeepSeek time", deepseek_time)
    json.dump(
        {"qwen": qwen_results, "deepseek": deepseek_results},
        open("match_results-1.5B.json", "w"),
    )


if __name__ == "__main__":
    main()
