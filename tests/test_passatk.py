import json
import os
import sys

sys.path.append("/Users/jiyi/Documents/2025-Projects/yj-verl")
from verl.evaluation.math_utils import REWARD_REGISTRY, pass_at_k

tmp = json.load(open("match_results.json"))

# for k in [1, 8]:
#     res = pass_at_k(tmp["qwen"], k)
#     print(f"pass@{k} for qwen: {res}")


def get_all_pass_k(all_results):
    pass_at_ks = {}
    for k in [1, 2, 4, 5, 8, 10]:
        res = pass_at_k(all_results, k)
        if res is not None:
            pass_at_ks[f"pass@{k}"] = res
    return pass_at_ks


qwenpassk = get_all_pass_k(tmp["qwen"])
deepseekpassk = get_all_pass_k(tmp["deepseek"])

print(json.dumps(qwenpassk, indent=2))
print("===" * 20)
print(json.dumps(deepseekpassk, indent=2))
