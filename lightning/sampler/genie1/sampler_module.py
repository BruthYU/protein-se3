import os
import sys
import torch
import argparse
import numpy as np
from tqdm import tqdm, trange
from lightning.model.genie1.lightning_model import genie1_Lightning_Model
from omegaconf import DictConfig, OmegaConf
import logging
from lightning.data.genie2.feat_utils import save_np_features_to_pdb
torch.set_float32_matmul_precision('high')



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

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



    def run_sampling(self):
        # sanity check
        min_length = self.infer_conf.min_n_res
        max_length = self.infer_conf.max_n_res

        for length in trange(min_length, max_length + 1):
            for batch_idx in range(self.infer_conf.num_batches):
                mask = torch.cat([
                    torch.ones((self.infer_conf.batch_size, length)),
                    torch.zeros((self.infer_conf.batch_size, self.conf.dataset.max_n_res - length))
                ], dim=1).to(self.device)
                ts = self.lightning_module.model.p_sample_loop(mask, self.infer_conf.noise_scale, verbose=False)[-1]
                for batch_sample_idx in range(ts.shape[0]):
                    sample_idx = batch_idx * self.infer_conf.batch_size + batch_sample_idx
                    coords = ts[batch_sample_idx].trans.detach().cpu().numpy()
                    coords = coords[:length]
                    np_features = {'atom_positions': coords}
                    output_pdb_filepath = os.path.join(
                        self.infer_conf.output_dir, 'pdbs',
                        '{}.pdb'.format(f'len_{length}_idx_{sample_idx}')
                    )
                    save_np_features_to_pdb(np_features, output_pdb_filepath)
