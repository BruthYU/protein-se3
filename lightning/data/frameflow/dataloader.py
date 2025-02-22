import pdb

import numpy as np
from preprocess.tools import so3_utils, chemical, residue_constants, protein
import torch
import logging
import torch.distributed as dist
import math
Protein = protein.Protein
import warnings

def _read_clusters(cluster_path):
    pdb_to_cluster = {}
    with open(cluster_path, "r") as f:
        for i,line in enumerate(f):
            for chain in line.split(' '):
                pdb = chain.split('_')[0]
                pdb_to_cluster[pdb.upper()] = i
    return pdb_to_cluster


class NewBatchSampler:

    def __init__(
            self,
            *,
            data_conf,
            dataset,
            seed=123,
            shuffle=True,
            num_replicas=None,
            rank=None,
            is_training = True,
            sample_mode = "cluster_length_batch"
    ):
        super().__init__()
        self._log = logging.getLogger(__name__)
        if num_replicas is None:
            self.num_replicas = dist.get_world_size()
        else:
            self.num_replicas = num_replicas
        if rank is None:
            self.rank = dist.get_rank()
        else:
            self.rank = rank

        self._data_conf = data_conf
        self._data_csv = dataset.csv
        self._is_training = is_training
        self._sample_mode = sample_mode

        # Each replica needs the same number of batches. We set the number
        # of batches to arbitrarily be the number of examples per replica.
        self._num_batches = self.get_num_batches()

        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0
        self.max_batch_size =  self._data_conf.sampler.max_batch_size
        self._log.info(f'Created dataloader rank {self.rank+1} out of {self.num_replicas}')

    def get_num_batches(self):
        # Each replica needs the same number of batches. We set the number
        # of batches to arbitrarily be the number of examples per replica.
        if self._is_training:
            if 'cluster' in self._sample_mode:
                num_batches = self._data_csv['cluster'].nunique()
            else:
                num_batches = len(self._data_csv)
            num_batches = math.ceil(num_batches / self.num_replicas)
        else:
            num_batches = self._data_conf.num_eval_lengths
            spel = self._data_conf.samples_per_eval_length
            if spel % self.num_replicas:
                warnings.warn(f"sampler_per_eval_length ({spel}) is not divisible by num_devices ({self.num_replicas})."
                              f"Set sampler_per_eval_length to {math.ceil(spel/self.num_replicas)}")
        return num_batches



    def get_train_sample_order(self):
        rng = torch.Generator()
        rng.manual_seed(self.seed + self.epoch)
        if 'cluster' in self._data_csv:
            cluster_sample = self._data_csv.groupby('cluster').sample(
                1, random_state=self.seed + self.epoch)
            indices = cluster_sample['index'].tolist()
        else:
            indices = self._data_csv['index'].tolist()

        if self.shuffle:
            new_order = torch.randperm(len(indices), generator=rng).numpy().tolist()
            indices = [indices[i] for i in new_order]

        # csv on each replica
        if len(self._data_csv) > self.num_replicas:
            replica_csv = self._data_csv.iloc[
                indices[self.rank::self.num_replicas]
            ]
        else:
            replica_csv = self._data_csv

        # Each batch contains multiple proteins of the same length.
        sample_order = []
        for seq_len, len_df in replica_csv.groupby('modeled_seq_len'):
            max_batch_size = min(
                self.max_batch_size,
                self._data_conf.sampler.max_num_res_squared // seq_len ** 2 + 1,
            )
            num_batches = math.ceil(len(len_df) / max_batch_size)
            for i in range(num_batches):
                batch_df = len_df.iloc[i * max_batch_size:(i + 1) * max_batch_size]
                batch_indices = batch_df['index'].tolist()
                batch_repeats = math.floor(max_batch_size / len(batch_indices))
                sample_order.append(batch_indices * batch_repeats)

        # Remove any length bias (shuffle batch lists).
        new_order = torch.randperm(len(sample_order), generator=rng).numpy().tolist()

        sample_order = [sample_order[i] for i in new_order]
        return sample_order

    def get_eval_sample_order(self):
        # filter proteins based on lengths
        if self._data_conf.max_eval_length is None:
            all_lengths = self._data_csv.modeled_seq_len
        else:
            all_lengths = self._data_csv.modeled_seq_len[
                self._data_csv.modeled_seq_len <= self._data_conf.max_eval_length]

        length_indices = (len(all_lengths) - 1) * np.linspace(
            0.0, 1.0, self._data_conf.num_eval_lengths)
        length_indices = length_indices.astype(int)
        eval_lengths = all_lengths[length_indices]
        eval_csv = self._data_csv[self._data_csv.modeled_seq_len.isin(eval_lengths)]
        eval_csv = eval_csv.sort_values('modeled_seq_len', ascending=False)

        # get global csv
        replica_per_length \
            = max(1, math.ceil(self._data_conf.samples_per_eval_length / self.num_replicas))
        global_csv = eval_csv.groupby('modeled_seq_len').sample(
            replica_per_length * self.num_replicas,
            replace=True,
            random_state=self.epoch
        )
        indices = global_csv['index'].tolist()

        # csv on each replica
        assert len(self._data_csv) > self.num_replicas
        replica_csv = self._data_csv.iloc[
            indices[self.rank::self.num_replicas]
        ]


        # Each batch contains multiple proteins of the same length.
        sample_order = []
        for i in range(self._num_batches):
            batch_df = replica_csv.iloc[i * replica_per_length:(i + 1) * replica_per_length]
            batch_indices = batch_df['index'].tolist()
            sample_order.append(batch_indices)

        return sample_order



    def _replica_epoch_batches(self):
        # Make sure all replicas share the same seed on each epoch.

        if self._is_training:
            sample_order = self.get_train_sample_order()
        else:
            sample_order = self.get_eval_sample_order()

        return sample_order

    def _create_batches(self):
        # Make sure all replicas have the same number of batches Otherwise leads to bugs.
        # See bugs with shuffling https://github.com/Lightning-AI/lightning/issues/10947
        all_batches = []
        num_augments = -1
        while len(all_batches) < self._num_batches:
            all_batches.extend(self._replica_epoch_batches())
            num_augments += 1
            if num_augments > 1000:
                raise ValueError('Exceeded number of augmentations.')
        if len(all_batches) >= self._num_batches:
            all_batches = all_batches[:self._num_batches]
        self.sample_order = all_batches
        # print(f"----Sample Order: {self.sample_order}---")

    def __iter__(self):
        self._create_batches()
        self.epoch += 1
        return iter(self.sample_order)

    def __len__(self):
        return len(self.sample_order)