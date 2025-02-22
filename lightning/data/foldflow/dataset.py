import json
import lmdb
import pickle
from torch.utils import data
import tree
import torch
import numpy as np
import lightning.data.foldflow.dataloader as du
import pandas as pd
import os
import math
import random
import logging
import ot as pot
from functools import partial

from lightning.data.foldflow.rigid_helpers import assemble_rigid_mat, extract_trans_rots_mat
from evaluate.openfold.utils import rigid_utils
from scipy.spatial.transform import Rotation
from lightning.data.foldflow.so3_helpers import so3_relative_angle
import lightning.data.foldflow.se3_fm as se3_fm

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

class foldflow_Dataset(data.Dataset):
    def __init__(self,
                 lmdb_cache,
                 data_conf = None,
                 fm_conf = None,
                 is_training= True,
                 is_OT = False,
                 ot_fn = "exact",
                 reg = 0.05,
                 max_same_res = 10
                 ):
        super().__init__()
        assert lmdb_cache, "No cache to build dataset."
        self.lmdb_cache = lmdb_cache
        self.csv = self.lmdb_cache.csv
        self.data_conf = data_conf
        self.fm_conf = fm_conf
        self.is_training = is_training

        # Could be Diffusion, CFM, OT-CFM or SF2M
        self.gen_model = se3_fm.SE3FlowMatcher(self.fm_conf)
        self.is_OT = is_OT
        self.reg = reg
        self.max_same_res = self.get_max_same_res(max_same_res)
        self.ot_fn = self.get_ot_fn(ot_fn)


    def get_ot_fn(self, ot_fn):
        # import ot as pot
        if ot_fn == "exact":
            return pot.emd
        elif ot_fn == "sinkhorn":
            return partial(pot.sinkhorn, reg=self.reg)

    def get_max_same_res(self, max_same_res):
        if self.is_OT:
            return max_same_res
        else:
            return -1

    def __len__(self):
        return len(self.csv)


    def __getitem__(self, idx):
        # Custom sampler can return None for idx None.
        # Hacky way to simulate a fixed batch size.
        if idx is None:
            return None

        # print(f"[DEBUG] Train dataset getitem")

        # Use a fixed seed for evaluation.
        if self.is_training:
            rng = np.random.default_rng(None)
        else:
            rng = np.random.default_rng(idx)

        chain_feats, gt_bb_rigid, pdb_name, csv_row = self.lmdb_cache.get_cache_csv_row(idx)

        if self.is_training and not self.is_OT:
            # Sample t and flow.
            t = rng.uniform(self.data_conf.min_t, 1.0)
            gen_feats_t = self.gen_model.forward_marginal(
                rigids_0=gt_bb_rigid, t=t, flow_mask=None, rigids_1=None
            )
        elif self.is_training and self.is_OT:
            t = rng.uniform(self.data_conf.min_t, 1.0)
            n_res = chain_feats["aatype"].shape[
                0
            ]  # feat['aatype'].shape = (batch, n_res)
            # get a maximum of self.max_same_res proteins with the same length
            subset = self.csv[self.csv["modeled_seq_len"] == n_res]
            n_samples = min(subset.shape[0], self.max_same_res)
            if n_samples == 1 or n_samples == 0:
                # only one sample, we can't do OT
                # self._log.info(f"Only one sample of length {n_res}, skipping OT")
                gen_feats_t = self.gen_model.forward_marginal(
                    rigids_0=gt_bb_rigid, t=t, flow_mask=None, rigids_1=None
                )
            else:
                sample_subset = subset.sample(
                    n_samples, replace=True, random_state=0
                ).reset_index(drop=True)

                # get the features, transform them to Rigid, and extract their translation and rotation.
                list_feat = [
                    self.lmdb_cache.get_cache_csv_row(i, sample_subset)[0] for i in range(n_samples)
                ]
                list_trans_rot = [
                    extract_trans_rots_mat(
                        rigid_utils.Rigid.from_tensor_7(feat["rigids_0"])
                    )
                    for feat in list_feat
                ]
                list_trans, list_rot = zip(*list_trans_rot)

                # stack them and change them to torch.tensor
                sample_trans = torch.stack(
                    [torch.from_numpy(trans) for trans in list_trans]
                )
                sample_rot = torch.stack([torch.from_numpy(rot) for rot in list_rot])

                device = sample_rot.device  # TODO: set the device before that...

                # random matrices on S03.
                rand_rot = torch.tensor(
                    Rotation.random(n_samples * n_res).as_matrix()
                ).to(device=device, dtype=sample_rot.dtype)
                rand_rot = rand_rot.reshape(n_samples, n_res, 3, 3)
                # rand_rot_axis_angle = matrix_to_axis_angle(rand_rot)

                # random translation
                rand_trans = torch.randn(size=(n_samples, n_res, 3)).to(
                    device=device, dtype=sample_trans.dtype
                )

                # compute the ground cost for OT: sum of the cost for S0(3) and R3.
                ground_cost = torch.zeros(n_samples, n_samples).to(device)

                for i in range(n_samples):
                    for j in range(i, n_samples):
                        s03_dist = torch.sum(
                            so3_relative_angle(sample_rot[i], rand_rot[j])
                        )
                        r3_dist = torch.sum(
                            torch.linalg.norm(sample_trans[i] - rand_trans[j], dim=-1)
                        )
                        ground_cost[i, j] = s03_dist**2 + r3_dist**2
                        ground_cost[j, i] = ground_cost[i, j]

                # OT with uniform distributions over the set of pdbs
                a = pot.unif(n_samples, type_as=ground_cost)
                b = pot.unif(n_samples, type_as=ground_cost)
                T = self.ot_fn(
                    a, b, ground_cost
                )  # NOTE: `ground_cost` is the squared distance on SE(3)^N.

                # sample using the plan
                # pick one random indices for the pdb returned by __getitem__
                idx_target = torch.randint(n_samples, (1,))
                pi_target = T[idx_target].squeeze()
                pi_target /= torch.sum(pi_target)
                idx_source = torch.multinomial(pi_target, 1)
                paired_rot = rand_rot[idx_source].squeeze()
                paired_trans = rand_trans[idx_source].squeeze()

                rigids_1 = assemble_rigid_mat(paired_rot, paired_trans)

                gen_feats_t = self.gen_model.forward_marginal(
                    rigids_0=gt_bb_rigid, t=t, flow_mask=None, rigids_1=rigids_1
                )

        else:
            t = 1.0
            gen_feats_t = self.gen_model.sample_ref(
                n_samples=gt_bb_rigid.shape[0],
                impute=gt_bb_rigid,
                flow_mask=None,
                as_tensor_7=True,
            )
        chain_feats.update(gen_feats_t)
        chain_feats["t"] = t

        # Convert all features to tensors.
        final_feats = tree.map_structure(
            lambda x: x if torch.is_tensor(x) else torch.tensor(x), chain_feats
        )
        final_feats = du.pad_feats(final_feats, csv_row["modeled_seq_len"])
        final_feats['lmdbIndex'] = idx
        return final_feats







