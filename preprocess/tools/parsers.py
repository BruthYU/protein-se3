"""Library for parsing different data structures."""
from Bio.PDB.Chain import Chain
import numpy as np
import torch
from preprocess.tools import residue_constants, protein
import json
from itertools import compress
Protein = protein.Protein



def process_chain(chain: Chain, chain_id: str) -> Protein:
    """Convert a PDB chain object into a AlphaFold Protein instance.
    
    Forked from alphafold.common.protein.from_pdb_string
    
    WARNING: All non-standard residue types will be converted into UNK. All
        non-standard atoms will be ignored.
    
    Took out lines 94-97 which don't allow insertions in the PDB.
    Sabdab uses insertions for the chothia numbering so we need to allow them.
    
    Took out lines 110-112 since that would mess up CDR numbering.
    
    Args:
        chain: Instance of Biopython's chain class.
    
    Returns:
        Protein object with protein features.
    """
    atom_positions = []
    aatype = []
    atom_mask = []
    residue_index = []
    b_factors = []
    chain_ids = []
    for res in chain:
        res_shortname = residue_constants.restype_3to1.get(res.resname, 'X')
        restype_idx = residue_constants.restype_order.get(
            res_shortname, residue_constants.restype_num)
        pos = np.zeros((residue_constants.atom_type_num, 3))
        mask = np.zeros((residue_constants.atom_type_num,))
        res_b_factors = np.zeros((residue_constants.atom_type_num,))
        for atom in res:
            if atom.name not in residue_constants.atom_types:
                continue
            pos[residue_constants.atom_order[atom.name]] = atom.coord
            mask[residue_constants.atom_order[atom.name]] = 1.
            res_b_factors[residue_constants.atom_order[atom.name]
                          ] = atom.bfactor
        aatype.append(restype_idx)
        atom_positions.append(pos)
        atom_mask.append(mask)
        residue_index.append(res.id[1])
        b_factors.append(res_b_factors)
        chain_ids.append(chain_id)

    return Protein(
        atom_positions=np.array(atom_positions),
        atom_mask=np.array(atom_mask),
        aatype=np.array(aatype),
        residue_index=np.array(residue_index),
        chain_index=np.array(chain_ids),
        b_factors=np.array(b_factors))

def process_chain_jsonl(entry, chain_id: str) -> Protein:
    """Convert a jsonl line into a AlphaFold Protein instance."""

    res_shortnames = list(residue_constants.restype_1to3.keys())


    atom_positions = []
    aatype = []
    atom_mask = []
    residue_index = []
    b_factors = []
    chain_ids = []

    atom_keys = []
    for key, val in entry['coords'].items():
        atom_keys.append(key)
        entry['coords'][key] = np.asarray(val)

    # get mask for filtering the nan values
    stacked_pos = np.stack([entry['coords'][key] for key in atom_keys], axis=1)
    X = torch.from_numpy(stacked_pos).float()
    nan_mask = ~torch.isnan(X.sum(dim=(1, 2)))

    for key in atom_keys:
        entry['coords'][key] =  entry['coords'][key][nan_mask]
    entry['seq'] = ''.join(list(compress(entry['seq'], nan_mask))),

    # data for AlphaFold Protein instance
    for idx, seq_shortname in enumerate(entry['seq'][0]):
        res_shortname = 'X' if seq_shortname not in res_shortnames else seq_shortname
        restype_idx = residue_constants.restype_order.get(
            res_shortname, residue_constants.restype_num)

        pos = np.zeros((residue_constants.atom_type_num, 3))
        mask = np.zeros((residue_constants.atom_type_num,))
        res_b_factors = np.zeros((residue_constants.atom_type_num,))
        for atom_key in atom_keys:
            if atom_key not in residue_constants.atom_types:
                continue
            pos[residue_constants.atom_order[atom_key]] = entry['coords'][atom_key][idx]
            mask[residue_constants.atom_order[atom_key]] = 1.

        aatype.append(restype_idx)
        atom_positions.append(pos)
        atom_mask.append(mask)
        residue_index.append(idx+1)
        b_factors.append(res_b_factors)
        chain_ids.append(chain_id)

    return Protein(
        atom_positions=np.array(atom_positions),
        atom_mask=np.array(atom_mask),
        aatype=np.array(aatype),
        residue_index=np.array(residue_index),
        chain_index=np.array(chain_ids),
        b_factors=np.array(b_factors))




