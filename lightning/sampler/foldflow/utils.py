import torch
import numpy as np
from evaluate.openfold.data import data_transforms


def process_chain(design_pdb_feats):
    chain_feats = {
        "aatype": torch.tensor(design_pdb_feats["aatype"]).long(),
        "all_atom_positions": torch.tensor(design_pdb_feats["atom_positions"]).double(),
        "all_atom_mask": torch.tensor(design_pdb_feats["atom_mask"]).double(),
    }
    chain_feats = data_transforms.atom37_to_frames(chain_feats)
    chain_feats = data_transforms.make_atom14_masks(chain_feats)
    chain_feats = data_transforms.make_atom14_positions(chain_feats)
    chain_feats = data_transforms.atom37_to_torsion_angles()(chain_feats)
    seq_idx = (
        design_pdb_feats["residue_index"]
        - np.min(design_pdb_feats["residue_index"])
        + 1
    )
    chain_feats["seq_idx"] = seq_idx
    chain_feats["res_mask"] = design_pdb_feats["bb_mask"]
    chain_feats["residue_index"] = design_pdb_feats["residue_index"]
    return chain_feats


def create_pad_feats(pad_amt):
    return {
        "res_mask": torch.ones(pad_amt),
        "fixed_mask": torch.zeros(pad_amt),
        "rigids_impute": torch.zeros((pad_amt, 4, 4)),
        "torsion_impute": torch.zeros((pad_amt, 7, 2)),
    }