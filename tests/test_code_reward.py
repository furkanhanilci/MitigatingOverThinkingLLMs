import unittest
from verl.utils.reward_score import code
from datasets import load_dataset

compute_score = code.compute_score
prime_data = load_dataset("PRIME-RL/Eurus-2-RL-Data")["train"]
prime_data = prime_data.filter(lambda x: "numina" not in x["data_source"])


class CodeRewardTest(unittest.TestCase):
    def test_reward_score(self):
        pass


if __name__ == "__main__":
    unittest.main9
