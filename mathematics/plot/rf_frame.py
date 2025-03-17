import numpy as np
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

    plt.savefig('a.pdf')
    fig.show()

def viz_frames(rigids, mask, ax=None, scale_factor=0.5, title='', ax_lim=8):
    viz_mask = mask.astype(bool)
    frames = rigids[viz_mask]
    se3_vec = rigids_to_se3_vec(frames, scale_factor=scale_factor)
    plot_se3(se3_vec, ax=ax, title=title, ax_lim=ax_lim)


if __name__ == '__main__':
    lmdb_cache = LMDB_Cache("../../preprocess/.cache/jsonl")
    csv = lmdb_cache.csv
    sampled_orders = csv[csv['modeled_seq_len']==100]
    chain_feats, _, pdb_names, _ = lmdb_cache.get_cache_csv_row(sampled_orders.index[0])
    print(pdb_names)
    frames = chain_feats['rigids_0']
    mask = chain_feats['res_mask']
    viz_frames(frames, mask)
