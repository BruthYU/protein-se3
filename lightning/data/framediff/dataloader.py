import numpy as np
from typing import List, Dict, Any
import collections
from preprocess.tools import so3_utils, chemical, residue_constants, protein
from evaluate.openfold.utils import rigid_utils
import string
from torch.utils import data
import torch
import logging
import random
import torch.distributed as dist
import math
from typing import Optional

Protein = protein.Protein

# Global map from chain characters to integers.
ALPHANUMERIC = string.ascii_letters + string.digits + ' '
CHAIN_TO_INT = {
    chain_char: i for i, chain_char in enumerate(ALPHANUMERIC)
}
INT_TO_CHAIN = {
    i: chain_char for i, chain_char in enumerate(ALPHANUMERIC)
}

CHAIN_FEATS = [
    'atom_positions', 'aatype', 'atom_mask', 'residue_index', 'b_factors'
]
UNPADDED_FEATS = [
    't', 'rot_score_scaling', 'trans_score_scaling', 't_seq', 't_struct', 'lmdbIndex'
]
RIGID_FEATS = [
    'rigids_0', 'rigids_t'
]
PAIR_FEATS = [
    'rel_rots'
]


move_to_np = lambda x: x.cpu().detach().numpy()
aatype_to_seq = lambda aatype: ''.join([
        residue_constants.restypes_with_x[x] for x in aatype])


def pad_feats(raw_feats, max_len, use_torch=False):
    padded_feats = {
        feat_name: pad(feat, max_len, use_torch=use_torch)
        for feat_name, feat in raw_feats.items()
        if feat_name not in UNPADDED_FEATS + RIGID_FEATS
    }
    for feat_name in PAIR_FEATS:
        if feat_name in padded_feats:
            padded_feats[feat_name] = pad(padded_feats[feat_name], max_len, pad_idx=1)
    for feat_name in UNPADDED_FEATS:
        if feat_name in raw_feats:
            padded_feats[feat_name] = raw_feats[feat_name]
    for feat_name in RIGID_FEATS:
        if feat_name in raw_feats:
            padded_feats[feat_name] = pad_rigid(raw_feats[feat_name], max_len)
    return padded_feats

def pad_rigid(rigid: torch.tensor, max_len: int):
    num_rigids = rigid.shape[0]
    pad_amt = max_len - num_rigids
    pad_rigid = rigid_utils.Rigid.identity(
        (pad_amt,), dtype=rigid.dtype, device=rigid.device, requires_grad=False)
    return torch.cat([rigid, pad_rigid.to_tensor_7()], dim=0)

def pad(x: np.ndarray, max_len: int, pad_idx=0, use_torch=False, reverse=False):
    """Right pads dimension of numpy array.

    Args:
        x: numpy like array to pad.
        max_len: desired length after padding
        pad_idx: dimension to pad.
        use_torch: use torch padding method instead of numpy.

    Returns:
        x with its pad_idx dimension padded to max_len
    """
    # Pad only the residue dimension.
    seq_len = x.shape[pad_idx]
    pad_amt = max_len - seq_len
    pad_widths = [(0, 0)] * x.ndim
    if pad_amt < 0:
        raise ValueError(f'Invalid pad amount {pad_amt}')
    if reverse:
        pad_widths[pad_idx] = (pad_amt, 0)
    else:
        pad_widths[pad_idx] = (0, pad_amt)
    if use_torch:
        return torch.pad(x, pad_widths)
    return np.pad(x, pad_widths)

# read A3M and convert letters into
# integers in the 0..20 range,
# also keep track of insertions



def concat_np_features(
        np_dicts: List[Dict[str, np.ndarray]], add_batch_dim: bool):
    """Performs a nested concatenation of feature dicts.

    Args:
        np_dicts: list of dicts with the same structure.
            Each dict must have the same keys and numpy arrays as the values.
        add_batch_dim: whether to add a batch dimension to each feature.

    Returns:
        A single dict with all the features concatenated.
    """
    combined_dict = collections.defaultdict(list)
    for chain_dict in np_dicts:
        for feat_name, feat_val in chain_dict.items():
            if add_batch_dim:
                feat_val = feat_val[None]
            combined_dict[feat_name].append(feat_val)
    # Concatenate each feature
    for feat_name, feat_vals in combined_dict.items():
        combined_dict[feat_name] = np.concatenate(feat_vals, axis=0)
    return combined_dict

def length_batching(
        np_dicts: List[Dict[str, np.ndarray]],
        max_squared_res: int,
    ):
    get_len = lambda x: x['res_mask'].shape[0]
    dicts_by_length = [(get_len(x), x) for x in np_dicts]
    length_sorted = sorted(dicts_by_length, key=lambda x: x[0], reverse=True)
    max_len = length_sorted[0][0]
    max_batch_examples = int(max_squared_res // max_len**2)
    pad_example = lambda x: pad_feats(x, max_len)
    padded_batch = [
        pad_example(x) for (_, x) in length_sorted[:max_batch_examples]]
    return torch.utils.data.default_collate(padded_batch)

def create_data_loader(
        torch_dataset: data.Dataset,
        batch_size,
        shuffle,
        sampler=None,
        num_workers=0,
        np_collate=False,
        max_squared_res=1e6,
        length_batch=False,
        drop_last=False,
        prefetch_factor=2):
    """Creates a data loader with jax compatible data structures."""
    if np_collate:
        collate_fn = lambda x: concat_np_features(x, add_batch_dim=True)
    elif length_batch:
        collate_fn = lambda x: length_batching(
            x, max_squared_res=max_squared_res)
    else:
        collate_fn = None
    persistent_workers = True if num_workers > 0 else False
    prefetch_factor = 2 if num_workers == 0 else prefetch_factor
    return data.DataLoader(
        torch_dataset,
        sampler=sampler,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        drop_last=drop_last,
        # Need fork https://github.com/facebookresearch/hydra/issues/964
        multiprocessing_context='fork' if num_workers != 0 else None,
        )


class TrainSampler(data.Sampler):

    def __init__(
            self,
            *,
            data_conf,
            dataset,
            batch_size,
            sample_mode,
            ):
        if not dist.is_available():
            raise RuntimeError("Requires distributed package to be available")
        self.num_replicas = dist.get_world_size()
        assert batch_size % self.num_replicas == 0, "Batch size must be divisible by num_gpus"

        self._log = logging.getLogger(__name__)
        self._data_conf = data_conf
        self._dataset = dataset
        self._data_csv = self._dataset.csv.reset_index(drop=True)
        self._dataset_indices = list(range(len(self._data_csv)))
        self._data_csv['index'] = self._dataset_indices
        self._batch_size = batch_size
        self.epoch = 0
        self._sample_mode = sample_mode
        self.sampler_len = len(self._dataset_indices)
        self.sample_list = None

        if self._sample_mode in ['cluster_length_batch', 'cluster_time_batch']:
            self._pdb_to_cluster = self._read_clusters()
            self._max_cluster = max(self._pdb_to_cluster.values())
            self._log.info(f'Read {self._max_cluster} clusters.')
            self._missing_pdbs = 0
            def cluster_lookup(pdb):
                pdb = pdb.split(".")[0].upper()
                if pdb not in self._pdb_to_cluster:
                    self._pdb_to_cluster[pdb] = self._max_cluster + 1
                    self._max_cluster += 1
                    self._missing_pdbs += 1
                return self._pdb_to_cluster[pdb]
            self._data_csv['cluster'] = self._data_csv['pdb_name'].map(cluster_lookup)
            num_clusters = len(set(self._data_csv['cluster']))
            self.sampler_len = num_clusters * self._batch_size
            self._log.info(
                f'Training on {num_clusters} clusters. PDBs without clusters: {self._missing_pdbs}'
            )

    def _read_clusters(self):
        pdb_to_cluster = {}
        with open(self._data_conf.cluster_path, "r") as f:
            for i,line in enumerate(f):
                for chain in line.split(' '):
                    pdb = chain.split('_')[0]
                    pdb_to_cluster[pdb.upper()] = i
        return pdb_to_cluster

    def __iter__(self):
        if self._sample_mode == 'length_batch':
            # Each batch contains multiple proteins of the same length.
            sampled_order = self._data_csv.groupby('modeled_seq_len').sample(
                self._batch_size, replace=True, random_state=self.epoch)
            self.sample_list = sampled_order
            return iter(sampled_order['index'].tolist())
        elif self._sample_mode == 'time_batch':
            # Each batch contains multiple time steps of the same protein.
            random.shuffle(self._dataset_indices)
            repeated_indices = np.repeat(self._dataset_indices, self._batch_size)
            self.sample_list = repeated_indices
            return iter(repeated_indices)
        elif self._sample_mode == 'cluster_length_batch':
            # Each batch contains multiple clusters of the same length.
            sampled_clusters = self._data_csv.groupby('cluster').sample(
                1, random_state=self.epoch)
            sampled_order = sampled_clusters.groupby('modeled_seq_len').sample(
                self._batch_size, replace=True, random_state=self.epoch)
            self.sample_list = sampled_order
            return iter(sampled_order['index'].tolist())
        elif self._sample_mode == 'cluster_time_batch':
            # Each batch contains multiple time steps of a protein from a cluster.
            sampled_clusters = self._data_csv.groupby('cluster').sample(
                1, random_state=self.epoch)
            dataset_indices = sampled_clusters['index'].tolist()
            repeated_indices = np.repeat(dataset_indices, self._batch_size)
            self.sample_list = repeated_indices
            return iter(repeated_indices.tolist())
        else:
            raise ValueError(f'Invalid sample mode: {self._sample_mode}')

    def set_epoch(self, epoch):
        self.epoch = epoch
    def __len__(self):
        return self.sampler_len

class NewDistributedSampler(data.Sampler):
    def __init__(self,
                 *,
                 data_conf,
                 dataset,
                 batch_size,
                 sample_mode,
                 num_replicas: Optional[int] = None,
                 rank: Optional[int] = None,
                 seed: int = 0, drop_last: bool = False, is_training: bool = True) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        self._log = logging.getLogger(__name__)
        self._data_conf = data_conf
        self._dataset = dataset
        self._batch_size = batch_size
        self._sample_mode = sample_mode
        self._data_csv = self._dataset.csv
        self._dataset_indices = list(range(len(self._data_csv)))
        self._data_csv['index'] = self._dataset_indices
        self._is_training = is_training

        if self._sample_mode in ['cluster_length_batch', 'cluster_time_batch']:
            self._pdb_to_cluster = self._read_clusters()
            self._max_cluster = max(self._pdb_to_cluster.values())
            self._log.info(f'Read {self._max_cluster} clusters.')
            self._missing_pdbs = 0

            def cluster_lookup(pdb):
                pdb = pdb.upper()
                if pdb not in self._pdb_to_cluster:
                    self._pdb_to_cluster[pdb] = self._max_cluster + 1
                    self._max_cluster += 1
                    self._missing_pdbs += 1
                return self._pdb_to_cluster[pdb]

            self._data_csv['cluster'] = self._data_csv['pdb_name'].map(cluster_lookup)
            num_clusters = len(set(self._data_csv['cluster']))
            self.sampler_len = num_clusters * self._batch_size
            self._log.info(
                f'Training on {num_clusters} clusters. PDBs without clusters: {self._missing_pdbs}'
            )

        # Distributed Sample Setting
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        assert batch_size % num_replicas == 0, "Batch size must be divisible by num_gpus"



        self.drop_last = drop_last
        if self._is_training:
            start_sample_list = self.get_train_sample_list()
        else:
            start_sample_list = self.get_eval_sample_list()
        # _repeated_size is the size of the dataset multiply by batch size
        self._repeated_size = len(start_sample_list)
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and self._repeated_size % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (self._repeated_size - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(self._repeated_size / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas
        self.seed = seed

    def _read_clusters(self):
        pdb_to_cluster = {}
        with open(self._data_conf.cluster_path, "r") as f:
            for i, line in enumerate(f):
                for chain in line.split(' '):
                    pdb = chain.split('_')[0]
                    pdb_to_cluster[pdb.upper()] = i
        return pdb_to_cluster

    def get_train_sample_list(self):
        if self._sample_mode == 'length_batch':
            # Each batch contains multiple proteins of the same length.
            sampled_order = self._data_csv.groupby('modeled_seq_len').sample(
                self._batch_size, replace=True, random_state=self.epoch)
            return sampled_order['index'].tolist()
        elif self._sample_mode == 'time_batch':
            # Each batch contains multiple time steps of the same protein.
            random.shuffle(self._dataset_indices)
            repeated_indices = np.repeat(self._dataset_indices, self._batch_size)
            return repeated_indices
        elif self._sample_mode == 'cluster_length_batch':
            # Each batch contains multiple clusters of the same length.
            sampled_clusters = self._data_csv.groupby('cluster').sample(
                1, random_state=self.epoch)
            sampled_order = sampled_clusters.groupby('modeled_seq_len').sample(
                self._batch_size, replace=True, random_state=self.epoch)
            return sampled_order['index'].tolist()
        elif self._sample_mode == 'cluster_time_batch':
            # Each batch contains multiple time steps of a protein from a cluster.
            sampled_clusters = self._data_csv.groupby('cluster').sample(
                1, random_state=self.epoch)
            dataset_indices = sampled_clusters['index'].tolist()
            repeated_indices = np.repeat(dataset_indices, self._batch_size)
            return repeated_indices
        else:
            raise ValueError(f'Invalid sample mode: {self._sample_mode}')

    def get_eval_sample_list(self):
        '''
        eval_num = num_eval_lengths * self._batch_size
        '''
        all_lengths = np.sort(self._data_csv.modeled_seq_len.unique())
        length_indices = (len(all_lengths) - 1) * np.linspace(
            0.0, 1.0, self._data_conf.num_eval_lengths)
        length_indices = length_indices.astype(int)
        eval_lengths = all_lengths[length_indices]
        eval_csv = self._data_csv[self._data_csv.modeled_seq_len.isin(eval_lengths)]
        # Fix a random seed to get the same split each time.
        eval_csv = eval_csv.groupby('modeled_seq_len').sample(
            self._batch_size, replace=True, random_state=self.epoch)
        eval_csv = eval_csv.sort_values('modeled_seq_len', ascending=False)
        return eval_csv['index'].tolist()


    def __iter__(self):
        if self._is_training:
            indices = self.get_train_sample_list()
        else:
            indices = self.get_eval_sample_list()
        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size == 0:
                pass
            elif padding_size <= len(indices):
                indices = np.concatenate((indices, indices[:padding_size]), axis=0)
            else:
                indices = np.concatenate(
                    (indices, np.repeat(indices, math.ceil(padding_size / len(indices)))[:padding_size]), axis=0)

        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        self.epoch += 1
        return iter(indices)


    def __len__(self) -> int:
        return self.num_samples










