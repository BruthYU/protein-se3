import json
import lmdb
import pickle
from torch.utils import data
import tree
import torch
import numpy as np
import lightning_protein.data.rfdiffusion.dataloader as du
from lightning_protein.data.rfdiffusion.diffusion import Diffuser
import pandas as pd
import os
import math
import random
import logging
from omegaconf import OmegaConf
import lightning_protein.model.rfdiffusion.util as mu
import torch.nn.functional as F
def _process_chain_feats(chain_feats):
    xyz = chain_feats['atom14_pos'].float()
    res_plddt = chain_feats['b_factors'][:, 1]
    res_mask = torch.tensor(chain_feats['res_mask']).int()
    return {
        'aatype': chain_feats['aatype'],
        'xyz': xyz,
        'res_mask': res_mask,
        'chain_idx': chain_feats["chain_idx"],
        'res_idx': chain_feats["seq_idx"],
    }

def _add_plddt_mask(feats, plddt_threshold):
    feats['plddt_mask'] = torch.tensor(
        feats['res_plddt'] > plddt_threshold).int()

def _read_clusters(cluster_path):
    pdb_to_cluster = {}
    with open(cluster_path, "r") as f:
        for i,line in enumerate(f):
            for chain in line.split(' '):
                pdb = chain.split('_')[0]
                pdb_to_cluster[pdb.upper()] = i
    return pdb_to_cluster

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
        self.csv = self.csv.reset_index()
        self.csv.to_csv("lmdb_protein.csv", index=False)

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

class rfdiffusion_Dataset(data.Dataset):
    def __init__(self,
                 lmdb_cache,
                 task='inpainting',
                 data_conf= None,
                 diffuser_conf= None,
                 is_training= True):
        super().__init__()
        assert lmdb_cache, "No cache to build dataset."
        self.lmdb_cache = lmdb_cache
        self.csv = self.lmdb_cache.csv
        self.data_conf = data_conf
        self.diffuser_conf = diffuser_conf
        self.is_training = is_training
        self.task = task
        self.diffuser = Diffuser(**self.diffuser_conf)

        self._rng = np.random.default_rng(seed=self.data_conf.seed)
        self._pdb_to_cluster = _read_clusters(self.data_conf.cluster_path)
        self._max_cluster = max(self._pdb_to_cluster.values())
        self._missing_pdbs = 0

        def cluster_lookup(pdb):
            pdb = pdb.split(".")[0].upper()
            if pdb not in self._pdb_to_cluster:
                self._pdb_to_cluster[pdb] = self._max_cluster + 1
                self._max_cluster += 1
                self._missing_pdbs += 1
            return self._pdb_to_cluster[pdb]

        self.csv['cluster'] = self.csv['pdb_name'].map(cluster_lookup)
        self._all_clusters = dict(
            enumerate(self.csv['cluster'].unique().tolist()))
        self._num_clusters = len(self._all_clusters)


    def process_chain_feats(self, chain_feats):
        return _process_chain_feats(chain_feats)

    def _new_sample_scaffold_mask(self, feats, rng):
        num_res = feats['res_mask'].shape[0]
        min_motif_size = int(self.data_conf.min_motif_percent * num_res)
        max_motif_size = int(self.data_conf.max_motif_percent * num_res)

        # Sample the total number of residues that will be used as the motif.
        motif_n_res = self._rng.integers(
            low=min_motif_size,
            high=max_motif_size
        )

        # motif_n_seg = 1
        # if self.data_conf.contiguous_percent < random.random():
        #     motif_n_seg = rng.integers(low=1, high=self.data_conf.max_motif_n_seg)

        motif_n_seg = rng.integers(low=1, high=self.data_conf.max_motif_n_seg)

        # Sample motif segments
        indices = sorted(np.random.choice(motif_n_res - 1, motif_n_seg - 1, replace=False) + 1)
        indices = [0] + indices + [motif_n_res]
        motif_seg_lens = [indices[i + 1] - indices[i] for i in range(motif_n_seg)]

        # Generate motif mask
        segs = [''.join(['1'] * l) for l in motif_seg_lens]
        segs.extend(['0'] * (num_res - motif_n_res))
        random.shuffle(segs)
        motif_mask = np.array([int(elt) for elt in ''.join(segs)])
        scaffold_mask = 1 - motif_mask
        return torch.from_numpy(scaffold_mask) * feats['res_mask']

    def sample_timestep_t(self):
        # Indexed from 1 to T
        # In the evaluation mode, set t to the last time step
        t = self.diffuser_conf.T
        if self.is_training:
            t = self._rng.integers(low=1, high=self.diffuser_conf.T + 1)
        # return torch.ones(1, dtype=torch.long) * t
        return t

    def setup_inpainting(self, feats, rng):
        scaffold_mask = self._new_sample_scaffold_mask(feats, rng)
        if 'plddt_mask' in feats:
            scaffold_mask = scaffold_mask* feats['plddt_mask']
        feats['res_mask'] = scaffold_mask

    def __getitem__(self, idx):
        chain_feats, gt_bb_rigid, pdb_name, csv_row = self.lmdb_cache.get_cache_csv_row(idx)
        feats = self.process_chain_feats(chain_feats)
        feats['scaffold_mask'] = feats["res_mask"]

        if self.task == 'hallucination':
            feats["motif_mask"] = 1 - feats["res_mask"]
        elif self.task == 'inpainting':
            if self.data_conf.inpainting_percent < random.random():
                feats['motif_mask'] = 1 - feats['res_mask']
            else:
                rng = self._rng if self.is_training else np.random.default_rng(seed=123)
                self.setup_inpainting(feats, rng)
                feats["motif_mask"] = 1 - feats['res_mask']

        else:
            raise ValueError(f'Unknown task {self.task}')

        feats['t'] = self.sample_timestep_t()
        feats['input_seq_onehot'] = F.one_hot(feats['aatype'], num_classes=22)
        feats_xyz = torch.zeros((len(feats['xyz']), 14, 3))
        feats_xyz[:,:14,:] = feats['xyz']
        # fa_stack: (T, L, 14, 3) T = self.diffuser_conf.T
        feats['fa_stack'], feats['xyz_true'] = self.diffuser.diffuse_pose(feats_xyz, diffusion_mask=feats['motif_mask'].bool())
        # feats['fa_stack'], feats['xyz_true'] = fa_stack[:, :, :14], xyz_true

        # Storing the csv index is helpful for debugging.
        feats['lmdbIndex'] = torch.ones(1, dtype=torch.long) * idx
        return feats

if __name__ == '__main__':
    conf = OmegaConf.load('../../config/method/rfdiffusion.yaml')
    data_conf = conf.dataset
    diffuser_conf = conf.diffuser
    lmdb_cache = LMDB_Cache(data_conf)
    rf_dataset = rfdiffusion_Dataset(lmdb_cache, "inpainting", data_conf, diffuser_conf)
    feat_1 = rf_dataset[1]
    pass
