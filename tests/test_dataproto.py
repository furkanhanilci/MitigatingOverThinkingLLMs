import torch
import numpy as np


class DataProto:
    def __init__(self, batch, non_tensor_batch, meta_info):
        self.batch = batch
        self.non_tensor_batch = non_tensor_batch
        self.meta_info = meta_info

    @staticmethod
    def concat(data):
        """Concat a list of DataProto. The batch is concatenated among dim=0."""
        batch_lst = []
        for batch in data:
            batch_lst.append(batch.batch)  # Extract batch tensor

        if batch_lst[0] is not None:  # Assumes first element is not None
            new_batch = torch.cat(batch_lst, dim=0)  # ERROR can happen here
        else:
            new_batch = None

        non_tensor_batch = {key: np.concatenate([d.non_tensor_batch[key] for d in data], axis=0)
                            for key in data[0].non_tensor_batch}

        return DataProto(new_batch, non_tensor_batch, data[0].meta_info)


# **Case 1: Empty batch_lst**
data_samples = []  # No data at all
try:
    DataProto.concat(data_samples)
except Exception as e:
    print("Error (Case 1):", e)

# **Case 2: batch_lst contains an empty tensor**
data_samples = [
    DataProto(batch=torch.tensor([]), non_tensor_batch={
              'a': np.array([1])}, meta_info={}),
    DataProto(batch=torch.tensor([]), non_tensor_batch={
              'a': np.array([2])}, meta_info={})
]
try:
    DataProto.concat(data_samples)
except Exception as e:
    print("Error (Case 2):", e)

# **Case 3: batch_lst contains tensors with inconsistent shapes**
data_samples = [
    DataProto(batch=torch.randn(0, 5), non_tensor_batch={
              'a': np.array([1])}, meta_info={}),
    DataProto(batch=torch.randn(2, 5), non_tensor_batch={
              'a': np.array([2])}, meta_info={})
]
try:
    DataProto.concat(data_samples)
except Exception as e:
    print("Error (Case 3):", e)
