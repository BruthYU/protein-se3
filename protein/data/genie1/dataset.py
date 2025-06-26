import json
import lmdb
import pickle
from torch.utils import data
import tree
import torch
import numpy as np
from preprocess.tools.residue_constants import restypes_with_x
import pandas as pd
import os
import math
import random
import logging

from omegaconf import OmegaConf

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

    def get_cache_csv_row(self, idx):
        return self.chain_ftrs[idx]






class genie1_Dataset(data.Dataset):
    def __init__(self,
                 lmdb_cache,
                 data_conf=None,):
        super().__init__()
        assert lmdb_cache, "No cache to build dataset."
        self.lmdb_cache = lmdb_cache
        self.csv = self.lmdb_cache.csv
        self.data_conf = data_conf

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        chain_feats = self.lmdb_cache.get_cache_csv_row(idx)
        coords = chain_feats['bb_positions']
        n_res = len(coords)
        coords = np.concatenate([coords, np.zeros((self.data_conf.max_n_res - n_res, 3))], axis=0)
        mask = np.concatenate([np.ones(n_res), np.zeros(self.data_conf.max_n_res - n_res)])
        return coords, mask






