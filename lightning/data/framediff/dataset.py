import json
import lmdb
import pickle
from torch.utils import data
import tree
import torch
import numpy as np
import lightning.data.framediff.dataloader as du
from lightning.data.framediff import se3_diffuser
import pandas as pd
import os
import math
import random
import logging

class LMDB_Cache:
    def __init__(self, data_conf):
        self.local_cache = None
        self.csv = None
        self.cache_dir = data_conf.cache_dir
        self.cache_to_memory()

    def cache_to_memory(self):
        print(f"Loading cache from local dataset @ {self.cache_dir}")
        self.local_cache = lmdb.open(self.cache_dir)
        result_tuples = []
        with self.local_cache.begin() as txn:
            for _, value in txn.cursor():
                result_tuples.append(pickle.loads(value))

        '''
        Lmdb index may not match filtered_protein.csv due to multiprocessing,
        So we directly recover csv from the lmdb cache. 
        '''
        lmdb_series = [x[3] for x in result_tuples]
        self.csv = pd.DataFrame(lmdb_series).reset_index(drop=True)
        self.csv.to_csv("lmdb_protein.csv", index=True)

        def _get_list(idx):
            return list(map(lambda x: x[idx], result_tuples))
        self.chain_ftrs = _get_list(0)
        self.gt_bb_rigid_vals = _get_list(1)
        self.pdb_names = _get_list(2)
        self.csv_rows = _get_list(3)

    def get_cache_csv_row(self, idx):
        # if self.csv is not None:
        #     # We are going to get the idx row out of the csv -> so we look for true index based on index cl
        #     idx = self.csv.iloc[idx]["index"]

        return (
            self.chain_ftrs[idx],
            self.gt_bb_rigid_vals[idx],
            self.pdb_names[idx],
            self.csv_rows[idx],
        )



class framediff_Dataset(data.Dataset):
    def __init__(self,
                 lmdb_cache,
                 data_conf = None,
                 frame_conf = None,
                 is_training= True):
        super().__init__()
        assert lmdb_cache, "No cache to build dataset."
        self.lmdb_cache = lmdb_cache
        self.csv = self.lmdb_cache.csv
        self.data_conf = data_conf
        self.is_training = is_training
        self.diffuser = se3_diffuser.SE3Diffuser(frame_conf)


    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        # Use a fixed seed for evaluation.
        if self.is_training:
            rng = np.random.default_rng(None)
        else:
            rng = np.random.default_rng(idx)

        chain_feats, gt_bb_rigid, pdb_name, csv_row = self.lmdb_cache.get_cache_csv_row(idx)
        # print(f"idx: {idx}  | pdb_name: {pdb_name} | csv_len: {csv_row.seq_len} | modeled_seq_len: {csv_row.modeled_seq_len}")

        # Sample t and diffuse.
        if self.is_training:
            t = rng.uniform(self.data_conf.min_t, 1.0)
            diff_feats_t = self.diffuser.forward_marginal(
                rigids_0=gt_bb_rigid,
                t=t,
                diffuse_mask=None
            )
        else:
            t = 1.0
            diff_feats_t = self.diffuser.forward_marginal(
                rigids_0=gt_bb_rigid,
                t=t,
                diffuse_mask=None
            )
        chain_feats.update(diff_feats_t)
        chain_feats['t'] = t

        chain_feats = du.pad_feats(chain_feats, csv_row['modeled_seq_len'])
        chain_feats['lmdbIndex'] = idx
        # Convert all features to tensors.
        final_feats = tree.map_structure(
            lambda x: x if torch.is_tensor(x) else torch.tensor(x), chain_feats)




        return final_feats







if __name__ == '__main__':

    instance = framediff_Dataset()
