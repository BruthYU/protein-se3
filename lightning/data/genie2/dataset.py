import json
import lmdb
import pickle
from torch.utils import data
import tree
import torch
import numpy as np
import lightning.data.genie2.feat_utils as feat_utils
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






class genie2_Dataset(data.Dataset):
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
        aatype, atom_positions, chain_idx = \
            chain_feats['aatype'], chain_feats['bb_positions'], chain_feats['chain_idx']
        all_chain_idx = np.unique(chain_idx).tolist()
        lengths = [np.sum(chain_idx == x) for x in all_chain_idx]
        np_features = feat_utils.create_empty_np_features(lengths)

        aatype = aatype.numpy()

        np_features['aatype'] = np.eye(len(restypes_with_x))[aatype]
        np_features['atom_positions'] = atom_positions.astype(float)

        if np.random.random() <= self.data_conf.motif_prob:
            np_features = self._update_motif_masks(np_features)

        # Pad
        np_features = feat_utils.pad_np_features(
            np_features,
            self.data_conf.max_n_chain,
            self.data_conf.max_n_res
        )

        return np_features

    def _update_motif_masks(self, np_features):
        """
        Update fixed sequence and structure mask in the feature dictionary to indicate
        where to provide motif sequence and structure information as conditions. Note
        that since Genie 2 is trained on single-motif scaffolding tasks, we did not
        modify fixed_group in the feature dictionary since all motif residues belong to
        the same group (initialized to group 0).

        Implemention of Algorithm 1.

        Args:
            np_features:
                A feature dictionary containing information on an input structure
                of length N, including
                    -	aatype:
                            [N, 20] one-hot encoding on amino acid types
                    -	num_chains:
                            [1] number of chains in the structure
                    -	num_residues:
                            [1] number of residues in the structure
                    -	num_residues_per_chain:
                            [1] an array of number of residues by chain
                    -	atom_positions:
                            [N, 3] an array of Ca atom positions
                    -	residue_mask:
                            [N] residue mask to indicate which residue position is masked
                    -	residue_index:
                            [N] residue index (started from 0)
                    -	chain_index:
                            [N] chain index (started from 0)
                    -	fixed_sequence_mask:
                            [N] mask to indicate which residue contains conditional
                            sequence information
                    -	fixed_structure_mask:
                            [N, N] mask to indicate which pair of residues contains
                            conditional structural information
                    -	fixed_group:
                            [N] group index to indicate which group the residue belongs to
                            (useful for specifying multiple functional motifs)
                    -	interface_mask:
                            [N] deprecated and set to all zeros.

        Returns:
            np_features:
                An updated feature dictionary.
        """

        # Sanity check
        assert np_features['num_chains'] == 1, 'Input must be monomer'

        # Sample number of motif residues
        motif_n_res = np.random.randint(
            np.floor(np_features['num_residues'] * self.data_conf.motif_min_pct_res),
            np.ceil(np_features['num_residues'] * self.data_conf.motif_max_pct_res)
        )

        # Sample number of motif segments
        motif_n_seg = np.random.randint(
            self.data_conf.motif_min_n_seg,
            min(self.data_conf.motif_max_n_seg, motif_n_res) + 1
        )

        # Sample motif segments
        indices = sorted(np.random.choice(motif_n_res - 1, motif_n_seg - 1, replace=False) + 1)
        indices = [0] + indices + [motif_n_res]
        motif_seg_lens = [indices[i + 1] - indices[i] for i in range(motif_n_seg)]

        # Generate motif mask
        segs = [''.join(['1'] * l) for l in motif_seg_lens]
        segs.extend(['0'] * (np_features['num_residues'] - motif_n_res))
        random.shuffle(segs)
        motif_sequence_mask = np.array([int(elt) for elt in ''.join(segs)]).astype(bool)
        motif_structure_mask = motif_sequence_mask[:, np.newaxis] * motif_sequence_mask[np.newaxis, :]
        motif_structure_mask = motif_structure_mask.astype(bool)

        # Update
        np_features['fixed_sequence_mask'] = motif_sequence_mask
        np_features['fixed_structure_mask'] = motif_structure_mask

        return np_features




