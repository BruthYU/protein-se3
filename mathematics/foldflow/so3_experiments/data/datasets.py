"""Copyright (c) Dreamfold."""
from torch.utils.data import Dataset
import numpy as np
import torch
import os
import collections
from typing import List, Dict
from scipy.spatial.transform import Rotation
def concat_np_features(
        np_dicts: List[Dict[str, np.ndarray]], add_batch_dim: bool):
    """Performs a nested concatenation of feature dicts.

    Args:
        np_dicts: list of dicts with the same structure.
            Each dict must have the same keys and numpy arrays as the values.
        add_batch_dim: whether to add a batch dimension to each feature.

    Returns:
        A single dict with all the features concatenated.
    """
    combined_dict = collections.defaultdict(list)
    for chain_dict in np_dicts:
        for feat_name, feat_val in chain_dict.items():
            if add_batch_dim:
                feat_val = feat_val[None]
            combined_dict[feat_name].append(feat_val)
    # Concatenate each feature
    for feat_name, feat_vals in combined_dict.items():
        combined_dict[feat_name] = np.concatenate(feat_vals, axis=0)
    return combined_dict

def get_split(data, split, seed):
    assert split in ["train", "valid", "test", "all"], f"split {split} not supported."
    if split != "all":
        rng = np.random.default_rng(seed)
        indices = np.arange(len(data))
        rng.shuffle(indices)

        n = len(data)
        if split == "train":
            data = data[indices[: int(n * 0.8)]]
        elif split == "valid":
            data = data[indices[int(n * 0.8) : int(n * 0.9)]]
        elif split == "test":
            data = data[indices[int(n * 0.8) : ]]
            # data = data
    return data


class SpecialOrthogonalGroup(Dataset):
    def __init__(self, root="data", name="bunny_group.npy", split="train", seed=12345):
        data = np.load(f"{root}/{name}").astype("float32")
        self.data = get_split(data, split, seed)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class SDE_Dataset(Dataset):
    def __init__(self, root="data", name="bunny_group.npy", so3_diffuser = None, data_conf=None, split="train", seed=12345):
        data = np.load(f"{root}/{name}").astype("float32")
        self.so3_diffuser = so3_diffuser
        self.data = get_split(data, split, seed)
        self.data_conf = data_conf

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        rng = np.random.default_rng(idx)
        t = rng.uniform(self.data_conf.min_t, 1.0)
        rot_0 = self.data[idx][None]
        rot_0 = Rotation.from_matrix(rot_0).as_rotvec()
        rot_t, rot_score = self.so3_diffuser.forward_marginal(rot_0, t)
        rot_score_scaling = self.so3_diffuser.score_scaling(t)
        diff_feat_t = {"t": np.array([t]),
                       "rot_t": rot_t,
                       "rot_score": rot_score,
                       "rot_score_scaling": rot_score_scaling}
        return diff_feat_t


class DDPM_Dataset(Dataset):
    def __init__(self, root="data", name="bunny_group.npy", split="train", seed=12345):
        data = np.load(f"{root}/{name}").astype("float32")
        self.data = get_split(data, split, seed)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]