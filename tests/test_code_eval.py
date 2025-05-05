# This file contains codes for evaluating the code execution evaluation.

import json
import unittest
import multiprocessing
from datasets import load_dataset
from verl.evaluation.code_eval import check_correctness
from verl.evaluation.code_eval.apps.apps_utils import run_test

prime = load_dataset("PRIME-RL/Eurus-2-RL-Data")["validation"]
prime_subset = prime.filter(lambda x: "numina" not in x["data_source"])
bigcode = load_dataset("bigcode/bigcodebench")["v0.1.3"]


class TestCodeEval(unittest.TestCase):
    def _test_prime_sample(self):
        return False
        pass

    def test_app_sample(self):
        sample = prime_subset[0]

        tmpsample = {"input_output": json.loads(sample["reward_model"]["ground_truth"])}
        code = """import sys
from itertools import permutations

# Read input
def read_input():
    n, m = map(int, sys.stdin.readline().split())
    graph = [[float('inf')] * n for _ in range(n)]
    
    for _ in range(m):
        x, y, w = map(int, sys.stdin.readline().split())
        x -= 1
        y -= 1
        graph[x][y] = min(graph[x][y], w)
        graph[y][x] = min(graph[y][x], w)  # Since the graph is undirected
    
    return n, graph

# Solve using DP + Bitmasking
def tsp(n, graph):
    dp = [[float('inf')] * n for _ in range(1 << n)]
    dp[1][0] = 0  # Start from vertex 1 (index 0 in zero-based indexing)
    
    for mask in range(1 << n):
        for u in range(n):
            if not (mask & (1 << u)):
                continue
            for v in range(n):
                if (mask & (1 << v)) == 0 and graph[u][v] < float('inf'):
                    dp[mask | (1 << v)][v] = min(dp[mask | (1 << v)][v], dp[mask][u] + graph[u][v])
    
    final_mask = (1 << n) - 1
    min_cycle_length = float('inf')
    
    for v in range(1, n):  # Ending at any vertex and returning to 1
        if graph[v][0] < float('inf'):
            min_cycle_length = min(min_cycle_length, dp[final_mask][v] + graph[v][0])
    
    return min_cycle_length if min_cycle_length < float('inf') else -1

# Main Execution
if __name__ == "__main__":
    n, graph = read_input()
    print(tsp(n, graph))
"""

        def p_run_test(queue):
            result = run_test(
                problem=tmpsample,
                test=code,
                # debug=True,
                debug=False,
            )
            queue.put(result)

        queue = multiprocessing.Queue()
        process = multiprocessing.Process(target=p_run_test, args=(queue,))
        process.start()
        process.join()
        result = queue.get()
        print(result)
        print("Done running app correctness check")

    def test_prime_sample(self):
        sample = bigcode[0]
        tmp_solution = sample["complete_prompt"] + sample["canonical_solution"]

        def run_corrctness(queue):
            result = check_correctness(
                completion_id=0,
                sample=sample,
                solution=tmp_solution,
                max_as_limit=30 * 1024,
                max_data_limit=30 * 1024,
                max_stack_limit=10,
            )
            queue.put(result)

        queue = multiprocessing.Queue()
        process = multiprocessing.Process(target=run_corrctness, args=(queue,))
        process.start()
        process.join()
        result = queue.get()
        print(result)
        print("Done running prime correctness check")


if __name__ == "__main__":
    unittest.main()
