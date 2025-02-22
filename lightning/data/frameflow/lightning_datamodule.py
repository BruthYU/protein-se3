import math
import torch
import logging
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler, dist
import inspect
import importlib
import logging


LOG = logging.getLogger(__name__)

class frameflow_Lightning_Datamodule(pl.LightningDataModule):
    def __init__(self, conf):
        super().__init__()
        self.save_hyperparameters()
        self.conf = conf
        self.data_conf= conf.dataset
        self.method_name = conf.method_name
        self.sample_mode = conf.experiment.train_sample_mode
        self.data_module = self.init_data_module(self.method_name)
        self.cache_module = self.init_cache_module(self.method_name)
        # import utils for to create dataloader
        self.dataloader = importlib.import_module(f'lightning.data.{self.method_name}.dataloader')

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.lmdb_cache = self.instancialize_module(module=self.cache_module, data_conf=self.data_conf)
            '''Train Dataset & Sampler'''
            self.trainset = self.instancialize_module(module=self.data_module, task = self.data_conf.task, lmdb_cache=self.lmdb_cache,
                                                      is_training=True, data_conf=self.data_conf)

            '''Valid Dataset & Sampler'''
            self.valset = self.instancialize_module(module=self.data_module, task = self.data_conf.task, lmdb_cache=self.lmdb_cache,
                                                    is_training=False, data_conf=self.data_conf)

    def train_dataloader(self, rank=None, num_replicas=None):
        num_workers = self.data_conf.loader.num_workers
        train_sampler = self.dataloader.NewBatchSampler(
            data_conf=self.data_conf,
            dataset=self.trainset,
            is_training= True,
            sample_mode= self.sample_mode
        )
        return DataLoader(
            self.trainset,
            batch_sampler=train_sampler,
            num_workers=num_workers,
            prefetch_factor=None if num_workers == 0 else self.data_conf.loader.prefetch_factor,
            pin_memory=False,
            persistent_workers=True if num_workers > 0 else False,
        )

    def val_dataloader(self, rank=None, num_replicas=None):
        num_workers = self.data_conf.loader.num_workers
        valid_sampler = self.dataloader.NewBatchSampler(
            data_conf=self.data_conf,
            dataset=self.valset,
            is_training=False,
            sample_mode=None
        )
        return DataLoader(
            self.valset,
            batch_sampler=valid_sampler,
            num_workers=num_workers,
            prefetch_factor=None if num_workers == 0 else self.data_conf.loader.prefetch_factor,
            pin_memory=False,
            persistent_workers=True if num_workers > 0 else False,
        )


    def instancialize_module(self, module, **other_args):
        class_args = list(inspect.signature(module.__init__).parameters)[1:]
        inkeys = other_args.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = other_args[arg]
        args1.update(other_args)
        return module(**args1)

    def init_data_module(self, name, **other_args):
        return getattr(importlib.import_module(f'data.{name}.dataset'), f'{name}_Dataset')

    def init_cache_module(self, name, **other_args):
        return getattr(importlib.import_module(f'data.{name}.dataset'), f'LMDB_Cache')