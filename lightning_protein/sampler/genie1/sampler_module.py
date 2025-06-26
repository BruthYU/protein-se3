import os
import sys
import torch
import argparse
import numpy as np
from tqdm import tqdm, trange
from lightning_protein.model.genie1.lightning_model import genie1_Lightning_Model
from omegaconf import DictConfig, OmegaConf
import logging
from preprocess.tools.chemical import aa_123
torch.set_float32_matmul_precision('high')



def save_as_pdb(seq, coords, filename, ca_only=True):
    coords = coords - np.mean(coords, axis=0, keepdims=True)
    coords = np.around(coords, decimals=3)
    def pad_left(string, length):
        assert len(string) <= length
        return ' ' * (length - len(string)) + string

    def pad_right(string, length):
        assert len(string) <= length
        return string + ' ' * (length - len(string))

    atom_list = ['N', 'CA', 'C', 'O']
    with open(filename, 'w') as file:
        for i in range(coords.shape[0]):
            atom = 'CA' if ca_only else pad_right(atom_list[i % 4], 2)
            atom_idx = i + 1
            residue_idx = i + 1 if ca_only else i // 4 + 1
            residue_name = aa_123[seq.upper()[residue_idx - 1]]
            line = 'ATOM  ' + pad_left(str(atom_idx), 5) + '  ' + pad_right(atom, 3) + ' ' + \
                   residue_name + ' ' + 'A' + pad_left(str(residue_idx), 4) + ' ' + '   ' + \
                   pad_left(str(coords[i][0]), 8) + pad_left(str(coords[i][1]), 8) + pad_left(str(coords[i][2]), 8) + \
                   '     ' + '      ' + '   ' + '  ' + pad_left(atom[0], 2)
            file.write(line + '\n')


class genie1_Sampler:
    def __init__(self, conf: DictConfig):
        self.conf = conf
        self.exp_conf = conf.experiment
        self.infer_conf = conf.inference
        self.log = logging.getLogger(__name__)
        self.ckpt_path = self.infer_conf.ckpt_path
        self.output_dir = self.infer_conf.output_dir
        self.lightning_module = genie1_Lightning_Model.load_from_checkpoint(
            checkpoint_path=self.ckpt_path
        )
        self.lightning_module.eval()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



    def run_sampling(self):
        pdbs_dir = os.path.join(self.infer_conf.output_dir, 'pdbs')
        if not os.path.exists(pdbs_dir):
            os.makedirs(pdbs_dir)

        # sanity check
        min_length = self.infer_conf.min_n_res
        max_length = self.infer_conf.max_n_res

        for length in range(min_length, max_length + 1):
            for batch_idx in range(self.infer_conf.num_batches):
                mask = torch.cat([
                    torch.ones((self.infer_conf.batch_size, length)),
                    torch.zeros((self.infer_conf.batch_size, self.conf.dataset.max_n_res - length))
                ], dim=1).to(self.device)
                ts = self.lightning_module.p_sample_loop(mask, self.infer_conf.noise_scale, verbose=True)[-1]
                for batch_sample_idx in range(ts.shape[0]):
                    sample_idx = batch_idx * self.infer_conf.batch_size + batch_sample_idx
                    coords = ts[batch_sample_idx].trans.detach().cpu().numpy()
                    coords = coords[:length]
                    seq = 'A' * coords.shape[0]
                    output_pdb_filepath = os.path.join(
                        self.infer_conf.output_dir, 'pdbs',
                        '{}.pdb'.format(f'len_{length}_idx_{sample_idx}')
                    )
                    save_as_pdb(seq, coords, output_pdb_filepath)
