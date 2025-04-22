import random
import lmdb
import functools as fn
import os
import time
import math
import torch
import pandas as pd
import numpy as np
from multiprocessing import get_context
from multiprocessing.managers import SharedMemoryManager
import sys
sys.path.append('..')

from scipy.spatial.transform import Rotation
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import logging
import pickle
from tqdm import tqdm
import tree
from preprocess.tools import utils as du
from evaluate.openfold.utils import rigid_utils
from evaluate.openfold.data import data_transforms
import warnings

_BYTES_PER_PROTEIN = int(2*1e6)



def get_list_chunk_slices(lst, chunk_size):
    return [(i, i + chunk_size) for i in range(0, len(lst), chunk_size)]

def get_csv_rows_many(csv, shared_list, idx_slice):
    start_idx, end_idx = tuple(map(lambda x: min(x, len(csv)), idx_slice))
    for idx in tqdm(list(range(start_idx, end_idx))):
        row_tuple = get_csv_row(csv, idx)
        while len(row_tuple[0]['aatype']) != int(row_tuple[-1]['modeled_seq_len']):
            l1 = len(row_tuple[0]['aatype'])
            l2 = int(row_tuple[-1]['modeled_seq_len'])
            warnings.warn(f'Lengths ({l1} | {l2}) mismatch: Protein {row_tuple[-2]} has been ignored')
            resample_idx = random.choice(range(start_idx, end_idx))
            row_tuple = get_csv_row(csv, resample_idx)
        shared_list[idx] = pickle.dumps(row_tuple)

    print("Finished saving data to pickle")


def get_csv_row(csv, idx):
    """Get on row of the csv file, and prepare the pdb feature dict.

    Args:
        idx (int): idx of the row
        csv (pd.DataFrame): csv pd.DataFrame

    Returns:
        tuple: dict of the features, ground truth backbone rigid, pdb_name
    """
    # Sample data example.
    example_idx = idx
    csv_row = csv.iloc[example_idx]
    if "pdb_name" in csv_row:
        pdb_name = csv_row["pdb_name"]
    elif "chain_name" in csv_row:
        pdb_name = csv_row["chain_name"]
    else:
        raise ValueError("Need chain identifier.")
    processed_file_path = csv_row["processed_path"]
    chain_feats = _process_csv_row(csv, processed_file_path)

    gt_bb_rigid = rigid_utils.Rigid.from_tensor_4x4(chain_feats["rigidgroups_0"])[:, 0]
    flowed_mask = np.ones_like(chain_feats["res_mask"])
    if np.sum(flowed_mask) < 1:
        raise ValueError("At least one res could be diffused")
    fixed_mask = 1 - flowed_mask
    chain_feats["fixed_mask"] = fixed_mask
    chain_feats["rigids_0"] = gt_bb_rigid.to_tensor_7()
    chain_feats["sc_ca_t"] = torch.zeros_like(gt_bb_rigid.get_trans())

    return chain_feats, gt_bb_rigid, pdb_name, csv_row


'''
Numpy Version < 2.0
'''
def _process_csv_row(csv, processed_file_path):
    processed_feats = du.read_pkl(processed_file_path)
    processed_feats = du.parse_chain_feats(processed_feats)

    # Only take modeled residues.
    modeled_idx = processed_feats["modeled_idx"]
    min_idx = np.min(modeled_idx)
    max_idx = np.max(modeled_idx)
    del processed_feats["modeled_idx"]
    processed_feats = tree.map_structure(
        lambda x: x[min_idx : (max_idx + 1)], processed_feats
    )

    # Run through OpenFold data transforms.
    chain_feats = {
        "aatype": torch.tensor(processed_feats["aatype"]).long(),
        "all_atom_positions": torch.tensor(processed_feats["atom_positions"]).double(),
        "all_atom_mask": torch.tensor(processed_feats["atom_mask"]).double(),
    }
    chain_feats = data_transforms.atom37_to_frames(chain_feats)
    chain_feats = data_transforms.make_atom14_masks(chain_feats)
    chain_feats = data_transforms.make_atom14_positions(chain_feats)
    chain_feats = data_transforms.atom37_to_torsion_angles()(chain_feats)

    # Re-number residue indices for each chain such that it starts from 1.
    # Randomize chain indices.
    chain_idx = processed_feats["chain_index"]
    res_idx = processed_feats["residue_index"]
    new_res_idx = np.zeros_like(res_idx)
    new_chain_idx = np.zeros_like(res_idx)
    all_chain_idx = np.unique(chain_idx).tolist()
    shuffled_chain_idx = (
        np.array(random.sample(all_chain_idx, len(all_chain_idx)))
        - np.min(all_chain_idx)
        + 1
    )
    for i, chain_id in enumerate(all_chain_idx):
        chain_mask = (chain_idx == chain_id).astype(int)
        chain_min_idx = np.min(res_idx + (1 - chain_mask) * 1e3).astype(int)
        new_res_idx = new_res_idx + (res_idx - chain_min_idx + 1) * chain_mask

        # Shuffle chain_index
        replacement_chain_id = shuffled_chain_idx[i]
        new_chain_idx = new_chain_idx + replacement_chain_id * chain_mask

    # To speed up processing, only take necessary features
    final_feats = {
        "aatype": chain_feats["aatype"],
        "seq_idx": new_res_idx,
        "chain_idx": new_chain_idx,
        "residx_atom14_to_atom37": chain_feats["residx_atom14_to_atom37"],
        "residue_index": processed_feats["residue_index"],
        "res_mask": processed_feats["bb_mask"],
        "b_factors": processed_feats["b_factors"],
        "bb_positions": processed_feats["bb_positions"],
        "atom37_pos": chain_feats["all_atom_positions"],
        "atom37_mask": chain_feats["all_atom_mask"],
        "atom14_pos": chain_feats["atom14_gt_positions"],
        "atom14_mask": chain_feats["atom14_gt_exists"],
        "rigidgroups_0": chain_feats["rigidgroups_gt_frames"],
        "torsion_angles_sin_cos": chain_feats["torsion_angles_sin_cos"],
    }

    return final_feats


def _rog_quantile_curve(df, quantile, eval_x):
    y_quant = pd.pivot_table(
        df,
        values="radius_gyration",
        index="modeled_seq_len",
        aggfunc=lambda x: np.quantile(x, quantile),
    )
    x_quant = y_quant.index.to_numpy()
    y_quant = y_quant.radius_gyration.to_numpy()

    # Fit polynomial regressor
    poly = PolynomialFeatures(degree=4, include_bias=True)
    poly_features = poly.fit_transform(x_quant[:, None])
    poly_reg_model = LinearRegression()
    poly_reg_model.fit(poly_features, y_quant)

    # Calculate cutoff for all sequence lengths
    pred_poly_features = poly.fit_transform(eval_x[:, None])
    # Add a little more.
    pred_y = poly_reg_model.predict(pred_poly_features) + 0.1
    return pred_y


class BuildCache:
    def __init__(self, data_conf):

        self.data_conf = data_conf
        self.cache_path = data_conf.cache_path
        self._log = logging.getLogger(__name__)
        self._init_metadata()

    def _init_metadata(self):
        """Process metadata.csv with filtering configuration"""

        filter_conf = self.data_conf.filtering
        pdb_csv = pd.read_csv(self.data_conf.csv_path)
        self.all_n_rows = len(pdb_csv)

        if (
            filter_conf.allowed_oligomer is not None
            and len(filter_conf.allowed_oligomer) > 0 and pdb_csv.columns.__contains__("oligomeric_detail")
        ):
            pdb_csv = pdb_csv[
                pdb_csv.oligomeric_detail.isin(filter_conf.allowed_oligomer)
            ]

        if filter_conf.max_len is not None:
            print(f'max_len: {filter_conf.max_len}')
            pdb_csv = pdb_csv[pdb_csv.modeled_seq_len <= filter_conf.max_len]

        if filter_conf.min_len is not None:
            print(f'min_len: {filter_conf.min_len}')
            pdb_csv = pdb_csv[pdb_csv.modeled_seq_len >= filter_conf.min_len]

        if filter_conf.max_helix_percent is not None and pdb_csv.columns.__contains__("helix_percent"):
            print(f"max_helix_percent:{filter_conf.max_helix_percent}")
            pdb_csv = pdb_csv[pdb_csv.helix_percent < filter_conf.max_helix_percent]

        if filter_conf.max_loop_percent is not None and pdb_csv.columns.__contains__("coil_percent"):
            print(f"max_loop_percent:{filter_conf.max_loop_percent}")
            pdb_csv = pdb_csv[pdb_csv.coil_percent < filter_conf.max_loop_percent]

        if filter_conf.min_beta_percent is not None and pdb_csv.columns.__contains__("strand_percent"):
            print(f"min_beta_percent:{filter_conf.min_beta_percent}")
            pdb_csv = pdb_csv[pdb_csv.strand_percent > filter_conf.min_beta_percent]

        if filter_conf.rog_quantile is not None and filter_conf.rog_quantile > 0.0 \
                and pdb_csv.columns.__contains__("radius_gyration"):
            print(f"rog_quantile:{filter_conf.rog_quantile}")
            prot_rog_low_pass = _rog_quantile_curve(
                pdb_csv, filter_conf.rog_quantile, np.arange(filter_conf.max_len)
            )
            row_rog_cutoffs = pdb_csv.modeled_seq_len.map(
                lambda x: prot_rog_low_pass[x - 1]
            )
            pdb_csv = pdb_csv[pdb_csv.radius_gyration < row_rog_cutoffs]

        if filter_conf.subset is not None:
            pdb_csv = pdb_csv[: filter_conf.subset]

        pdb_csv = pdb_csv.sort_values("modeled_seq_len", ascending=False)
        pdb_csv = pdb_csv.reset_index(drop=True)
        self._create_split(pdb_csv)

    def _create_split(self, pdb_csv):
        # Training or validation specific logic.
        self.csv = pdb_csv
        self._log.info(f"Training: {len(self.csv)} examples")


    def _build_dataset_cache_v2(self):
        print(
            f"Starting to process dataset csv into memory "
        )
        print(f"Residue Length from {self.data_conf.filtering.min_len} to {self.data_conf.filtering.max_len}")
        print(f"All Rows {self.all_n_rows}")
        # self.csv = self.csv.iloc[:500]
        print(f"Running only {len(self.csv)}")


        build_local_cache = True
        if os.path.isdir(self.cache_path):
            raise ValueError(f"Found existing local cache dir @ {self.cache_path}, skipping build")

        os.makedirs(self.cache_path)

        filtered_csv_path = os.path.join(self.cache_path, "filtered_protein.csv")
        self.csv.to_csv(filtered_csv_path, index=False)

        # Initialize local cache with lmdb
        self._local_cache = lmdb.open(
            self.cache_path, map_size=(1024**3) * 20
        )  # 1GB * 5

        st_time = time.time()

        if build_local_cache:
            print(f"Building cache and saving @ {self.cache_path}")

            dataset_size = len(self.csv)
            num_chunks = math.ceil(
                float(dataset_size) / self.data_conf.num_csv_processors
            )

            idx_chunks = get_list_chunk_slices(list(range(dataset_size)), num_chunks)

            result_tuples = [None] * len(self.csv)

            pbar = tqdm(total=len(self.csv))
            with self._local_cache.begin(write=True) as txn:
                with SharedMemoryManager() as smm:
                    with get_context("spawn").Pool(
                        self.data_conf.num_csv_processors
                    ) as pool:
                        shared_list = smm.ShareableList(
                            [
                                bytes(_BYTES_PER_PROTEIN)
                                for _ in range(len(self.csv))
                            ]
                        )
                        partial_fxn = fn.partial(
                            get_csv_rows_many, self.csv, shared_list
                        )
                        iterator = enumerate(pool.imap(partial_fxn, idx_chunks))
                        for idx, _ in iterator:
                            start_idx, end_idx = tuple(
                                map(lambda x: min(x, len(self.csv)), idx_chunks[idx])
                            )
                            # print(f"RUNNING {start_idx} {end_idx} : chunks  {idx_chunks[idx]}")
                            for inner_idx in tqdm(range(start_idx, end_idx)):
                                txn.put(str(inner_idx).encode(), shared_list[inner_idx])
                                shared_list[inner_idx] = ""
                                pbar.update(1)

if __name__ == '__main__':
    from omegaconf import OmegaConf
    conf = OmegaConf.load('./config.yaml')
    cache_instance = BuildCache(conf)
    cache_instance._build_dataset_cache_v2()
    pass



