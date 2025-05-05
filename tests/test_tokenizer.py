import json
import torch
import timeit
import unittest
import random
import string
from dataclasses import dataclass
from datasets import load_dataset
from transformers import AutoTokenizer
from autotiktokenizer import AutoTikTokenizer
from typing import List, Any, Dict, Union

model_name = "Qwen/Qwen2.5-Math-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tiktokenizer = AutoTikTokenizer.from_pretrained(model_name)
# tmp = json.load(
#     open(
#         "/Users/jiyi/Library/Containers/com.tencent.xinWeChat/Data/Library/Application Support/com.tencent.xinWeChat/2.0b4.0.9/d18806e3ee98e838e2c4581b5e3cac9c/Message/MessageTemp/eea23f2c111d1c354850f828eb247d6a/File/error_file.json"
#     )
# )

ids = [2404] * 3000

special_tokens = [
    "<|im_end|>",
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<|object_ref_start|>",
    "<|object_ref_end|>",
    "<|box_start|>",
    "<|box_end|>",
    "<|quad_start|>",
    "<|quad_end|>",
    "<|vision_start|>",
    "<|vision_end|>",
    "<|vision_pad|>",
    "<|image_pad|>",
    "<|video_pad|>",
]


def origin_impl(ids) -> List[str]:
    tokens = tokenizer.convert_ids_to_tokens(
        # data.batch["responses"][i, :data.batch["response_length"][i]]
        ids
    )
    # print([tokenizer.convert_tokens_to_string([i]) for i in tokens if i is not None])
    # try:

    prefix_to_pos = {
        # len(
        tokenizer.convert_tokens_to_string(
            [tok for tok in tokens[:_i] if tok and tok not in special_tokens]
        ).strip():
        # ):
        # )
        _i
        for _i in range(len(tokens) + 1)
    }

    return prefix_to_pos


def tiktok_impl(ids) -> List[str]:
    # tokens =
    prefix_to_pos = {
        # len(tiktokenizer.decode(ids[:_i]).strip()): _i for _i in range(len(ids) + 1)
        tiktokenizer.decode(ids[:_i]).strip(): _i
        for _i in range(len(ids) + 1)
    }
    return prefix_to_pos


def random_string_with_newlines(length=20, newline_chance=0.2):
    """Generate a random string with random double newlines (\n\n) inserted."""
    characters = (
        string.ascii_letters + string.digits
    )  # Letters (upper & lower) + digits
    result = []

    for _ in range(length):
        if random.random() < newline_chance:  # Insert \n\n based on probability
            result.append("\n\n")
        result.append(random.choice(characters))

    return "".join(result)


@dataclass
class SampleConfig:
    split_method: str = "step"
    chunk_size: int = 1


@dataclass
class SampleData:
    batch: dict = None
    non_tensor_batch: dict = None


class TokenizerTest(unittest.TestCase):
    def init_test(self):
        model_name = "Qwen/Qwen2.5-Math-7B"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tiktokenizer = AutoTikTokenizer.from_pretrained(model_name)

    def test_split_reponse(self):
        self.init_test()
        bs = 10
        random_strings = [
            random_string_with_newlines(length=200, newline_chance=0.05)
            for _ in range(bs)
        ]
        data = SampleData(
            batch={
                "responses": self.tokenizer(
                    random_strings,
                    return_tensors="pt",
                    padding="longest",
                    padding_side="left",
                )["input_ids"],
                "response_length": torch.tensor([len(x) for x in random_strings]),
            },
            non_tensor_batch={
                "response_str": random_strings,
            },
        )

        for index in range(bs):
            self.config = SampleConfig(
                split_method="step",
                chunk_size=2,
            )

            # hf tokenizer result
            old_tiktokenizer = self.tiktokenizer
            self.tiktokenizer = None
            hf_res = self.split_response(data, index)

            # tiktoken tokenizer result
            self.tiktokenizer = old_tiktokenizer
            tik_res = self.split_response(data, index)

            # Assert the results are the same
            self.assertEqual(hf_res, tik_res)

    def test_split_timeit(self):
        self.init_test()
        bs = 10
        random_strings = [
            random_string_with_newlines(length=3000, newline_chance=0.05)
            for _ in range(bs)
        ]
        data = SampleData(
            batch={
                "responses": self.tokenizer(
                    random_strings,
                    return_tensors="pt",
                    padding="longest",
                    padding_side="left",
                )["input_ids"],
                "response_length": torch.tensor([len(x) for x in random_strings]),
            },
            non_tensor_batch={
                "response_str": random_strings,
            },
        )

        self.config = SampleConfig(
            split_method="step",
            chunk_size=2,
        )

        old_tiktokenizer = self.tiktokenizer
        self.tiktokenizer = None
        hf_execution_time = timeit.timeit(
            stmt=lambda: self.split_response(data, 0), globals=globals(), number=10
        )

        self.tiktokenizer = old_tiktokenizer
        tik_execution_time = timeit.timeit(
            stmt=lambda: self.split_response(data, 0), globals=globals(), number=10
        )
        print(f"hf: {hf_execution_time}")
        print(f"tik: {tik_execution_time}")

    def split_by_steps(
        self, solution: str, num_step_per_chunk: int = 1
    ) -> List[Dict[str, str]]:
        split_token = "\n\n"
        steps = solution.split(split_token)

        split_results = []
        for i in range(0, len(steps) + 1, num_step_per_chunk):
            partial_solution = split_token.join(steps[:i])
            split_results.append(partial_solution)
        if len(steps) % num_step_per_chunk != 0:
            # Add the last step
            split_results.append(split_token.join(steps))
        return split_results

    def split_response(self, data, index: int):
        def find_todo_steps(ids, special_tokens):
            res = [
                i
                for i, tok in enumerate(ids)
                if tok not in special_tokens and tok is not None and tok != ""
            ]
            return res

        if self.tiktokenizer is not None:
            if self.config.split_method == "step":
                partial_solutions = self.split_by_steps(
                    data.non_tensor_batch["response_str"][index],
                    num_step_per_chunk=self.config.chunk_size,
                )
                ids = (
                    data.batch["responses"][
                        index, : data.batch["response_length"][index]
                    ]
                    .cpu()
                    .tolist()
                )
                special_token_ids = [
                    self.tiktokenizer.encode(tok, allowed_special="all")
                    for tok in self.tiktokenizer.special_tokens_set
                ]
                todo_steps = find_todo_steps(ids, special_token_ids)
                prefix_to_pos = {
                    self.tiktokenizer.decode(ids[:_i])
                    .replace(self.tokenizer.pad_token, "")
                    .strip(): _i
                    # for _i in range(len(ids) + 1)
                    for _i in todo_steps
                }
                prefix_to_pos[""] = 0
                return [
                    {"partial_solution": s, "position": prefix_to_pos[s.strip()] - 1}
                    for s in partial_solutions
                ]
            elif self.config.split_method == "token":
                pass
        else:
            special_tokens = self.tokenizer.all_special_tokens
            if self.config.split_method == "step":
                partial_solutions = self.split_by_steps(
                    data.non_tensor_batch["response_str"][index],
                    num_step_per_chunk=self.config.chunk_size,
                )
                tokens = self.tokenizer.convert_ids_to_tokens(
                    data.batch["responses"][
                        index, : data.batch["response_length"][index]
                    ]
                )
                prefix_to_pos = {
                    self.tokenizer.convert_tokens_to_string(
                        [
                            tok
                            for tok in tokens[:_i]
                            if tok and tok not in special_tokens
                        ]
                    ).strip(): _i
                    for _i in range(len(tokens) + 1)
                }
                prefix_to_pos[""] = 0
                return [
                    {"partial_solution": s, "position": prefix_to_pos[s.strip()] - 1}
                    for s in partial_solutions
                ]

            elif self.config.split_method == "token":
                partial_solutions = [
                    {
                        "partial_solution": self.tokenizer.decode(
                            data.batch["responses"][index, :i], skip_special_tokens=True
                        ),
                        "position": i - 1,
                    }
                    for i in range(
                        0,
                        data.batch["response_length"][index] + 1,
                        self.config.chunk_size,
                    )
                ]
                if data.batch["response_length"][index] % self.config.chunk_size != 0:
                    partial_solutions.append(
                        {
                            "partial_solution": self.tokenizer.decode(
                                data.batch["responses"][
                                    index, : data.batch["response_length"][index]
                                ],
                                skip_special_tokens=True,
                            ),
                            "position": data.batch["response_length"][index] - 1,
                        }
                    )
                return partial_solutions


# def tik_impl(ids) -> List[str]:
#     strings = [
#         tiktokenizer.decode_single_token_bytes(i) if i < tiktokenizer.n_vocab else None
#         for i in ids
#     ]
#     print([x.decode(errors="ignore") for x in strings])

#     prefix = b""
#     pos = 0
#     res = {}
#     # res[""] = 0
#     res[0] = 0
#     for idx, string in enumerate(strings):
#         if (string is not None) and (
#             string.decode(errors="ignore") not in special_tokens
#         ):
#             prefix = prefix + string
#             pos = idx + 1
#             # res[prefix.decode(errors="ignore").strip()] = pos
#             res[len(prefix.decode(errors="ignore").strip())] = pos

#     return res


# def new_impl(ids) -> List[str]:
#     tokens = tokenizer.convert_ids_to_tokens(ids)
#     print(tokens)
#     # strings = [
#     #     tokenizer.convert_tokens_to_string([tokens[i]]) for i in range(len(tokens) + 1)
#     # ]
#     strings = [
#         tokenizer.convert_tokens_to_string([tokens[i]]).encode()
#         if (tokens[i] is not None and tokens[i] not in special_tokens)
#         else ""
#         for i in range(len(tokens))
#     ]
#     prefix = b""
#     pos = 0
#     res = {}
#     # res[""] = 0
#     res[0] = 0
#     for idx, string in enumerate(strings):
#         if string.decode(errors="ignore") != "":
#             prefix = prefix + string
#             pos = idx + 1
#             # res[prefix.strip()] = pos
#             res[len(prefix.decode(errors="ignore").strip())] = pos
#     return res


# def new_impl_old(ids) -> List[str]:
#     tokens = tokenizer.convert_ids_to_tokens(ids)
#     print(tokens)
#     # strings = [
#     #     tokenizer.convert_tokens_to_string([tokens[i]]) for i in range(len(tokens) + 1)
#     # ]
#     strings = [
#         tokenizer.convert_tokens_to_string([tokens[i]]).encode()
#         if (tokens[i] is not None and tokens[i] not in special_tokens)
#         else ""
#         for i in range(len(tokens))
#     ]
#     prefix = ""
#     pos = 0
#     res = {}
#     # res[""] = 0
#     res[0] = 0
#     for idx, string in enumerate(strings):
#         if string != "":
#             prefix = prefix + string
#             pos = idx + 1
#             # res[prefix.strip()] = pos
#             res[len(prefix.strip())] = pos
#     return res


def time_test():
    ids = [3044] * 3000
    origin_excution_time = timeit.timeit(
        "origin_impl(ids)", globals=globals(), number=10
    )
    new_execution_time = timeit.timeit("tiktok_impl(ids)", globals=globals(), number=10)
    print(f"origin: {origin_excution_time}")
    print(f"new: {new_execution_time}")


# origin = origin_impl(ids)
# new = tiktok_impl(ids)
# assert origin == new, f"failed at {l}"

# for l in [10, 20, 30, 50, 70]:
# origin = origin_impl(ids[:l])
# new = tiktok_impl(ids[:l])
# print(origin)
# print(new)
# assert origin == new, f"failed at {l}"

# l = 20

# # for l in [10, 13, 20, 30, 40, 50]:
# # for l in [15, 20, 30, 40, 50]:
# for l in [60, 70]:
#     # origin = origin_impl(tmp["ids"][:l])
#     # new = new_impl(tmp["ids"][:l])
#     origin = origin_impl(tmp["ids"][l : l + 10])
#     new = new_impl(tmp["ids"][l : l + 10])
#     # new = tik_impl(tmp["ids"][:l])

#     import ipdb

#     ipdb.set_trace()

#     print(origin)
#     print(new)
#     # import
#     print("Difference:")
#     # diffkey = set(origin.keys()) - set(new.keys())
#     # print({k: origin[k] for k in diffkey})

#     # diffkey = set(new.keys()) - set(origin.keys())
#     # print({k: origin[k] for k in diffkey})
#     # import ipdb

#     # ipdb.set_trace()
#     assert origin == new, f"failed at {l}"
#     print("---" * 20)

if __name__ == "__main__":
    unittest.main()
