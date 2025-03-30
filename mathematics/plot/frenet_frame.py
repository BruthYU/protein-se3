

import numpy as np
import torch
import geomstats.backend as gs
from geomstats.geometry.special_euclidean import SpecialEuclidean
import geomstats.visualization as visualization
import matplotlib.pyplot as plt
import lightning.data.framediff.dataloader as du
import lightning.model.framediff
from mpl_toolkits.mplot3d import Axes3D
import lmdb
import pickle
import pandas as pd
from scipy.spatial.transform import Rotation

n_samples = 10
translations = np.random.uniform(-2, 2, (n_samples, 3))  # Random translations
rot_vecs = np.random.uniform(-np.pi, np.pi, (n_samples, 3))  # Random rotation vectors
se3_points = np.hstack([rot_vecs, translations])  # Create SE(3) elements in Lie algebra


def compute_frenet_frames(coords, chains, mask, eps=1e-10):
    """
    Construct Frenet-Serret frames based on a sequence of coordinates.

    Since the Frenet-Serret frame is constructed based on three consecutive
    residues, for each chain, the rotational component of its first residue
    is assigned with the rotational component of its second residue; and the
    rotational component of its last residue is assigned with the rotational
    component of its second last residue.

    Args:
        coords:
            [B, N, 3] Per-residue atom positions.
        chains:
            [B, N] Per-residue chain indices.
        mask:
            [B, N] Residue mask.
        eps:
            Epsilon for computational stability. Default to 1e-10.

    Returns:
        rots:
            [B, N, 3, 3] Rotational components for the constructed frames.
    """

    # [B, N-1, 3]
    t = coords[:, 1:] - coords[:, :-1]
    t_norm = torch.sqrt(eps + torch.sum(t ** 2, dim=-1))
    t = t / t_norm.unsqueeze(-1)

    # [B, N-2, 3]
    b = torch.cross(t[:, :-1], t[:, 1:])
    b_norm = torch.sqrt(eps + torch.sum(b ** 2, dim=-1))
    b = b / b_norm.unsqueeze(-1)

    # [B, N-2, 3]
    n = torch.cross(b, t[:, 1:])

    # [B, N-2, 3, 3]
    tbn = torch.stack([t[:, 1:], b, n], dim=-1)

    # Construct rotation matrices
    rots = []
    for i in range(mask.shape[0]):
        rots_ = torch.eye(3).unsqueeze(0).repeat(mask.shape[1], 1, 1)
        length = torch.sum(mask[i]).int()
        rots_[1:length-1] = tbn[i, :length-2]

        # Handle start of chain
        for j in range(length):
            if j == 0 or chains[i][j] != chains[i][j-1]:
                rots_[j] = rots_[j+1]

        # Handle end of chain
        for j in range(length):
            if j == length - 1 or chains[i][j] != chains[i][j+1]:
                rots_[j] = rots_[j-1]

        # Update
        rots.append(rots_)

    # [B, N, 3, 3]
    rots = torch.stack(rots, dim=0).to(coords.device)

    return rots

class LMDB_Cache:
    def __init__(self, cache_dir):
        self.local_cache = None
        self.csv = None
        self.cache_dir = cache_dir
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


def rigids_to_se3_vec(frame, scale_factor=1.0):
    trans = frame[:, 4:] * scale_factor
    rotvec = Rotation.from_quat(frame[:, :4]).as_rotvec()
    se3_vec = np.concatenate([rotvec, trans], axis=-1)
    return se3_vec


def plot_se3(se3_vec, ax_lim=15, title=None, ax=None):
    if ax is None:
        fig = plt.figure(figsize=(20, 20))
        ax = fig.add_subplot(111, projection="3d")
    visualization.plot(se3_vec, ax=ax, space="SE3_GROUP")
    bb_trans = se3_vec[:, 3:]
    ln = ax.plot(bb_trans[:,0], bb_trans[:,1], bb_trans[:,2], alpha=1, c='gray')
    if ax_lim is not None:
        ax.set_xlim(-ax_lim, ax_lim)
        ax.set_ylim(-ax_lim, ax_lim)
        ax.set_zlim(-ax_lim, ax_lim)
    ax.set_axis_off()
    # ax.view_init(azim= 30, elev= 20)

    plt.savefig('figs/b.pdf')
    fig.show()

def viz_frames(frames, mask, ax=None, scale_factor=0.5, title='', ax_lim=8):
    viz_mask = mask.astype(bool)
    frames = frames[viz_mask]

    plot_se3(frames, ax=ax, title=title, ax_lim=ax_lim)


if __name__ == '__main__':
    lmdb_cache = LMDB_Cache("../../preprocess/.cache/jsonl")
    csv = lmdb_cache.csv
    sampled_orders = csv[csv['modeled_seq_len']==100]
    chain_feats, _, pdb_names, _ = lmdb_cache.get_cache_csv_row(sampled_orders.index[0])
    print(pdb_names)


    trans = chain_feats['rigids_0'][:,4:] * 0.5
    chain_idx = chain_feats['chain_idx']
    mask = torch.from_numpy(chain_feats['res_mask'])
    rots = compute_frenet_frames(trans[None,:], chain_idx[None,:], mask[None,:]).squeeze()
    rotvec = Rotation.from_matrix(rots).as_rotvec()
    frames = np.concatenate([rotvec, trans.squeeze().numpy()], axis=-1)

    viz_frames(frames, mask.numpy())
