import inspect
import importlib
import pytorch_lightning as pl
import random
import torch
from torch.utils.data import DataLoader
import logging


LOG = logging.getLogger(__name__)


class rfdiffusion_Lightning_Datamodule(pl.LightningDataModule):
    def __init__(self, conf):
        super().__init__()
        self.save_hyperparameters()
        self.conf = conf
        self.data_conf = conf.dataset
        self.exp_conf = conf.experiment
        self.diffuser_conf = conf.diffuser
        self.method_name = conf.method_name
        self.data_module = self.init_data_module(self.method_name)
        self.cache_module = self.init_cache_module(self.method_name)
        # import utils for to create dataloader
        self.dataloader = importlib.import_module(f'lightning.data.{self.method_name}.dataloader')



    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.lmdb_cache = self.instancialize_module(module=self.cache_module, data_conf=self.data_conf)
            '''Train Dataset & Sampler'''
            self.trainset = self.instancialize_module(module=self.data_module, lmdb_cache=self.lmdb_cache, is_training=True,
                                                      diffuser_conf=self.diffuser_conf, data_conf=self.data_conf)

            '''Valid Dataset & Sampler'''
            self.valset = self.instancialize_module(module=self.data_module, lmdb_cache=self.lmdb_cache, is_training=False,
                                                    diffuser_conf=self.diffuser_conf, data_conf=self.data_conf)

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