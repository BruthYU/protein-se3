import sys
sys.path.append('../../..')
print(sys.path)
import os
import glob
import torch
from tqdm import tqdm
import pandas as pd
from evaluate.pipeline.utils.parse import (
	parse_pdb_file,
)
from evaluate.pipeline.utils.secondary import (
	assign_secondary_structures,
	assign_left_handed_helices
)
from omegaconf import DictConfig, OmegaConf
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.colors import LinearSegmentedColormap

# 自定义一个白到红的渐变色图
white_red = LinearSegmentedColormap.from_list('white_red', ['white', 'red'])




class SecondaryStructure:
    def __init__(self, config: DictConfig):
        self.config = config
        self.workspace = self.config.inference.workspace
        self.pdbs_dir = os.path.join(self.workspace, 'pdbs')
        # self.designs_dir = os.path.join(self.workspace, 'design_all')
        self.results_dir = self.workspace

    def compute_secondary_structures(self):
        """
        Compute secondary diversity. Outputs are stored in the results directory, where each line
        in the file provides secondary structure statistics on a generated structure or its most
        similar structure predicted by the structure prediction model.

        Args:
            pdbs_dir:
                Directory containing PDB files for generated structures.
            designs_dir:
                Directory where each file is the most similar structure (predicted by the
                folding model) to the generated structure and is stored in a PDB format.
            results_dir:
                Result directory containing a file named 'single_scores.csv', where
                each line stores the self-consistency evaluation results on a generated
                structure.
        """

        # Create output filepath
        assert os.path.exists(self.results_dir), 'Missing output results directory'
        generated_secondary_filepath = os.path.join(self.results_dir, 'single_generated_secondary.csv')
        assert not os.path.exists(generated_secondary_filepath), 'Output generated secondary filepath existed'
        with open(generated_secondary_filepath, 'w') as file:
            columns = ['domain', 'generated_pct_helix', 'generated_pct_strand', 'generated_pct_ss',
                       'generated_pct_left_helix']
            file.write(','.join(columns) + '\n')


        # Process generated pdbs
        for generated_filepath in tqdm(
                glob.glob(os.path.join(self.pdbs_dir, '*.pdb')),
                desc='Computing generated secondary diversity'
        ):
            # Parse filepath
            domain = generated_filepath.split('/')[-1].split('.')[0]

            # Parse pdb file
            output = parse_pdb_file(generated_filepath)

            # Parse secondary structures
            ca_coords = torch.Tensor(output['ca_coords']).unsqueeze(0)
            pct_ss = torch.sum(assign_secondary_structures(ca_coords, full=False), dim=1).squeeze(0) / ca_coords.shape[
                1]
            pct_left_helix = torch.sum(assign_left_handed_helices(ca_coords).squeeze(0)) / ca_coords.shape[1]

            # Save
            with open(generated_secondary_filepath, 'a') as file:
                file.write('{},{:.3f},{:.3f},{:.3f},{:.3f}\n'.format(
                    domain, pct_ss[0], pct_ss[1], pct_ss[0] + pct_ss[1], pct_left_helix
                ))

        # Process designed pdbs
        # designed_secondary_filepath = os.path.join(self.results_dir, 'single_designed_secondary.csv')
        # assert not os.path.exists(designed_secondary_filepath), 'Output designed secondary filepath existed'
        # with open(designed_secondary_filepath, 'w') as file:
        #     columns = ['domain', 'designed_pct_helix', 'designed_pct_strand', 'designed_pct_ss',
        #                'designed_pct_left_helix']
        #     file.write(','.join(columns) + '\n')
        # for design_filepath in tqdm(
        #         glob.glob(os.path.join(self.designs_dir, '*.pdb')),
        #         desc='Computing designed secondary diversity'
        # ):
        #     # Parse filepath
        #     domain = design_filepath.split('/')[-1].split('.')[0]
        #
        #     # Parse pdb file
        #     output = parse_pdb_file(design_filepath)
        #
        #     # Parse secondary structures
        #     ca_coords = torch.Tensor(output['ca_coords']).unsqueeze(0)
        #     pct_ss = torch.sum(assign_secondary_structures(ca_coords, full=False), dim=1).squeeze(0) / ca_coords.shape[
        #         1]
        #     pct_left_helix = torch.sum(assign_left_handed_helices(ca_coords).squeeze(0)) / ca_coords.shape[1]
        #
        #     # Save
        #     with open(designed_secondary_filepath, 'a') as file:
        #         file.write('{},{:.3f},{:.3f},{:.3f},{:.3f}\n'.format(
        #             domain, pct_ss[0], pct_ss[1], pct_ss[0] + pct_ss[1], pct_left_helix
        #         ))

    def draw_heatmaps(self):
        generate_secondary_path = os.path.join(self.results_dir, 'single_generated_secondary.csv')
        generate_secondary_df = pd.read_csv(generate_secondary_path)
        design_secondary_path = os.path.join(self.results_dir, 'single_designed_secondary.csv')
        design_secondary_df = pd.read_csv(design_secondary_path)

        generate_alpha = generate_secondary_df['generated_pct_helix'].tolist()
        generate_beta = generate_secondary_df['generated_pct_strand'].tolist()
        design_alpha = design_secondary_df['designed_pct_helix'].tolist()
        design_beta = design_secondary_df['designed_pct_strand'].tolist()

        generate_points = np.array(list(zip(generate_alpha, generate_beta)))
        design_points = np.array(list(zip(design_alpha, design_beta)))

        # draw heatmap for generated samples
        x = generate_points[:, 0]
        y = generate_points[:, 1]
        grid_size = 0.1
        x_bins = np.arange(0, 1+grid_size, grid_size)
        y_bins = np.arange(0, 1+grid_size, grid_size)
        heatmap, xedges, yedges = np.histogram2d(x, y, bins=[x_bins, y_bins])
        # ax = sns.heatmap(heatmap.T/len(x), cmap='YlGnBu', vmin=0, vmax=0.1)
        # xticks = np.arange(0, 1, 0.2)
        # yticks = np.arange(0, 1, 0.2)
        #
        # ax.set_xticks(xticks)
        # ax.set_yticks(yticks)
        # plt.savefig(os.path.join(self.results_dir, 'generate_secondary_heatmap.png'))
        plt.imshow(heatmap.T/len(x), origin='lower',
                   extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                   vmin=0, vmax=0.5, cmap=white_red)
        plt.colorbar(label='Point Count')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title('Point Density Heatmap')
        plt.savefig(os.path.join(self.results_dir, 'generate_secondary_heatmap.png'))


        # draw heatmap for designed structures
        x = design_points[:, 0]
        y = design_points[:, 1]
        grid_size = 0.2
        x_bins = np.arange(0, 1+grid_size, grid_size)
        y_bins = np.arange(0, 1+grid_size, grid_size)
        heatmap, xedges, yedges = np.histogram2d(x, y, bins=[x_bins, y_bins])
        plt.imshow(heatmap.T, origin='lower',
                   extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                   )
        plt.colorbar(label='Point Count')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title('Point Density Heatmap')
        plt.savefig(os.path.join(self.results_dir, 'design_secondary_heatmap.png'))








if __name__ == '__main__':
    conf = OmegaConf.load('../../backup/config_standard.yaml')
    print('Starting inference')
    start_time = time.time()
    secondary_pipeline = SecondaryStructure(conf)
    secondary_pipeline.compute_secondary_structures()
    # secondary_pipeline.draw_heatmaps()
    elapsed_time = time.time() - start_time
    print(f'Finished in {elapsed_time:.2f}s')