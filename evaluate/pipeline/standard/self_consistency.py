import sys
sys.path.append('../../..')
sys.path.append('../../')
print(sys.path)
import os
import time
import tree
import numpy as np
import hydra
import torch
import subprocess
import logging
import pandas as pd
import shutil
from datetime import datetime
from biotite.sequence.io import fasta
import GPUtil
import random
import string
from typing import Optional


from lightning_protein.model.framediff.analysis import metrics
from preprocess.tools import utils as du
from preprocess.tools import residue_constants
from typing import Dict

from omegaconf import DictConfig, OmegaConf
import esm
os.environ['http_proxy'] = '127.0.0.1:7890'
os.environ['https_proxy'] = '127.0.0.1:7890'

CA_IDX = residue_constants.atom_order['CA']

class Pipeline:
    def __init__(self, conf: DictConfig):
        self._conf = conf

        self._infer_conf = conf.inference
        self._sample_conf = self._infer_conf.samples
        self._pmpnn_dir = self._infer_conf.pmpnn_dir
        self._workspace = self._infer_conf.workspace

        self.create_folders()

        # Set-up accelerator
        if torch.cuda.is_available():
            if self._infer_conf.gpu_id is None:
                available_gpus = ''.join(
                    [str(x) for x in GPUtil.getAvailable(
                        order='memory', limit = 8)])
                self.device = f'cuda:{available_gpus[0]}'
            else:
                self.device = f'cuda:{self._infer_conf.gpu_id}'
        else:
            self.device = 'cpu'

        self._folding_model = esm.pretrained.esmfold_v1().eval()
        self._folding_model = self._folding_model.to(self.device)

    def create_folders(self):
        #get reference pdb paths
        self.decoy_pdb_dir = os.path.join(self._workspace, 'pdbs')
        self.references = os.listdir(self.decoy_pdb_dir)
        self.references.sort()

        #create folders
        self.sequences_dir = os.path.join(self._workspace, 'sequences')
        assert not os.path.exists(self.sequences_dir), 'Output sequences directory existed'
        os.mkdir(self.sequences_dir)

        self.structures_dir = os.path.join(self._workspace, 'structures')
        assert not os.path.exists(self.structures_dir), 'Output structures directory existed'
        os.mkdir(self.structures_dir)

        self.scores_dir = os.path.join(self._workspace, 'scores')
        assert not os.path.exists(self.scores_dir), 'Output scores directory existed'
        os.mkdir(self.scores_dir)

        self.jsonl_dir = os.path.join(self._workspace, 'jsonl')
        assert not os.path.exists(self.jsonl_dir), 'Output scores directory existed'
        os.mkdir(self.jsonl_dir)

        self.designs_dir = os.path.join(self._workspace, 'designs')
        assert not os.path.exists(self.designs_dir ), 'Output sequences directory existed'
        os.mkdir(self.designs_dir )

        # if self._task == "scaffold":
        #     self.masks_dir = os.path.join(self._workspace, 'masks')
        #     assert os.path.exists(self.masks_dir), 'Masks directory doesn\'t exist'
        #     assert len(os.listdir(self.decoy_pdb_dir)) > 0, "Task name is scaffold, but there are no mask files"




    def run_self_consistency(self, clean=True):
        """Run self-consistency on design proteins against reference protein.

        Args:
            decoy_pdb_dir: directory where designed protein files are stored.
            reference_pdb_path: path to reference protein file
            motif_mask: Optional mask of which residues are the motif.

        Returns:
            Writes ProteinMPNN outputs to decoy_pdb_dir/seqs
            Writes ESMFold outputs to decoy_pdb_dir/esmf
            Writes results in decoy_pdb_dir/sc_results.csv
        """




        # Run PorteinMPNN
        output_path = os.path.join(self.jsonl_dir, f"parsed_pdbs.jsonl")
        process = subprocess.Popen([
            'python',
            f'{self._pmpnn_dir}/helper_scripts/parse_multiple_chains.py',
            f'--input_path={self.decoy_pdb_dir}',
            f'--output_path={output_path}',
        ])
        _ = process.wait()
        num_tries = 0
        ret = -1
        pmpnn_args = [
            'python',
            f'{self._pmpnn_dir}/protein_mpnn_run.py',
            '--out_folder',
            os.path.abspath(self.sequences_dir),
            '--jsonl_path',
            os.path.abspath(output_path),
            '--num_seq_per_target',
            str(self._sample_conf.seq_per_sample),
            '--sampling_temp',
            '0.1',
            '--seed',
            '38',
            '--batch_size',
            '1',
            '--ca_only'
        ]
        if self._infer_conf.gpu_id is not None:
            pmpnn_args.append('--device')
            pmpnn_args.append(str(self._infer_conf.gpu_id))
        while ret < 0:
            try:
                process = subprocess.Popen(
                    pmpnn_args,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.STDOUT
                )
                ret = process.wait()
            except Exception as e:
                num_tries += 1
                self._log.info(f'Failed ProteinMPNN. Attempt {num_tries}/5')
                torch.cuda.empty_cache()
                if num_tries > 4:
                    raise e
        # mpnn_fasta_path = os.path.join(
        #     self.decoy_pdb_dir,
        #     'seqs',
        #     os.path.basename(reference_pdb_path).replace('.pdb', '.fa')
        # )
        fa_names = [os.path.basename(reference_pdb_name).replace('.pdb', '.fa') for reference_pdb_name in self.references]
        mpnn_fasta_paths = [os.path.join(self.sequences_dir,'seqs',fa_name) for fa_name in fa_names]
        reference_pdb_paths = [os.path.join(self.decoy_pdb_dir, reference_pdb_name) for reference_pdb_name in self.references]


        for mpnn_idx, (reference_pdb_path, mpnn_fasta_path) in enumerate(zip(reference_pdb_paths, mpnn_fasta_paths)):
            # Run ESMFold on each ProteinMPNN sequence and calculate metrics.
            pure_pdb_name = self.references[mpnn_idx].split('.')[0]
            mpnn_results = {
                'pdb_name': [],
                'tm_score': [],
                'sample_path': [],
                'header': [],
                'sequence': [],
                'rmsd': [],
            }
            # if self._task == "scaffold":
            #     mpnn_results.update({'motif_rmsd':[]})

            # if motif_mask is not None:
            #     # Only calculate motif RMSD if mask is specified.
            #     mpnn_results['motif_rmsd'] = []
            esmf_dir = os.path.join(self.structures_dir, f'{mpnn_idx}')
            os.makedirs(esmf_dir, exist_ok=True)
            fasta_seqs = fasta.FastaFile.read(mpnn_fasta_path)
            sample_feats = du.parse_pdb_feats('sample', reference_pdb_path)
            for i, (header, string) in enumerate(fasta_seqs.items()):

                # Run ESMFold
                esmf_sample_path = os.path.join(esmf_dir, f'sample_{i}.pdb')
                _ = self.run_folding(string, esmf_sample_path)
                esmf_feats = du.parse_pdb_feats('folded_sample', esmf_sample_path)
                sample_seq = du.aatype_to_seq(sample_feats['aatype'])

                # Calculate scTM of ESMFold outputs with reference protein
                _, tm_score = metrics.calc_tm_score(
                    sample_feats['bb_positions'], esmf_feats['bb_positions'],
                    sample_seq, sample_seq)
                rmsd = metrics.calc_aligned_rmsd(
                    sample_feats['bb_positions'], esmf_feats['bb_positions'])

                # if self._task == "scaffold":
                #
                #     motif_mask_path = os.path.join(self.masks_dir, f"{pure_pdb_name}.npy")
                #     motif_mask = np.load(motif_mask_path)
                #     sample_motif = sample_feats['bb_positions'][motif_mask]
                #     of_motif = esmf_feats['bb_positions'][motif_mask]
                #     motif_rmsd = metrics.calc_aligned_rmsd(
                #         sample_motif, of_motif)
                #     mpnn_results['motif_rmsd'].append(motif_rmsd)

                mpnn_results['pdb_name'].append(pure_pdb_name)
                mpnn_results['rmsd'].append(rmsd)
                mpnn_results['tm_score'].append(tm_score)
                mpnn_results['sample_path'].append(esmf_sample_path)
                mpnn_results['header'].append(header)
                mpnn_results['sequence'].append(string)

            # Save results to CSV
            csv_path = os.path.join(self.scores_dir, f'sc_results_{mpnn_idx}.csv')
            mpnn_results = pd.DataFrame(mpnn_results)
            mpnn_results.to_csv(csv_path)
        self.aggregate_scores()
        if clean:
            shutil.rmtree(self.scores_dir)
            shutil.rmtree(self.structures_dir)
            shutil.rmtree(self.sequences_dir)
            shutil.rmtree(self.jsonl_dir)


    def run_folding(self, sequence, save_path):
        """Run ESMFold on sequence."""
        with torch.no_grad():
            output = self._folding_model.infer_pdb(sequence)

        with open(save_path, "w") as f:
            f.write(output)
        return output

    def aggregate_scores(self):
        # designs_name = ''.join(random.choices(string.ascii_letters, k=4))
        score_file_names = os.listdir(self.scores_dir)
        score_file_names.sort()
        score_file_paths = [os.path.join(self.scores_dir, name) for name in score_file_names]
        best_rows = []
        for idx, score_file_path in enumerate(score_file_paths):
            df = pd.read_csv(score_file_path)
            min_index = df['rmsd'].idxmin()
            min_row = df.loc[[min_index]]
            designs_name = min_row['pdb_name'].item()
            min_row['domain'] = f'design_{designs_name}'
            best_sample_path = min_row['sample_path'].item()
            best_rows.append(min_row)
            shutil.copy(best_sample_path, os.path.join(self.designs_dir,f'design_{designs_name}.pdb'))

        info_csv = pd.concat(best_rows, ignore_index=True)
        info_csv.to_csv(os.path.join(self._workspace, 'info.csv'), index=False)






if __name__ == '__main__':
    conf = OmegaConf.load('../../backup/config_standard.yaml')
    print('Starting inference')
    start_time = time.time()
    pipeline = Pipeline(conf)
    pipeline.run_self_consistency()
    elapsed_time = time.time() - start_time
    print(f'Finished in {elapsed_time:.2f}s')