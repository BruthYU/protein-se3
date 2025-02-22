import inspect
import importlib
import pytorch_lightning as pl
import random
import torch
from torch.utils.data import DataLoader
import logging

LOG = logging.getLogger(__name__)




class foldflow_Lightning_Datamodule(pl.LightningDataModule):
    def __init__(self, conf):
        super().__init__()
        self.save_hyperparameters()
        self.data_conf = conf.dataset
        self.exp_conf = conf.experiment
        self.fm_conf = conf.flow_matcher
        self.method_name = conf.method_name
        self.data_module = self.init_data_module(self.method_name)
        self.cache_module = self.init_cache_module(self.method_name)
        # import utils for to create dataloader
        self.dataloader = importlib.import_module(f'lightning.data.{self.method_name}.dataloader')


    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.lmdb_cache = self.instancialize_module(module=self.cache_module, data_conf=self.data_conf)
            '''Train Dataset'''
            self.trainset = self.instancialize_module(module=self.data_module, lmdb_cache=self.lmdb_cache, data_conf=self.data_conf, fm_conf=self.fm_conf,
                                                      is_training=True, is_OT=self.fm_conf.ot_plan, ot_fn=self.fm_conf.ot_fn, reg=self.fm_conf.reg)

            '''Valid Dataset'''
            self.valset = self.instancialize_module(module=self.data_module, lmdb_cache=self.lmdb_cache, data_conf=self.data_conf, fm_conf=self.fm_conf,
                                                      is_training=False, is_OT=self.fm_conf.ot_plan, ot_fn=self.fm_conf.ot_fn, reg=self.fm_conf.reg)



    def train_dataloader(self):
        # Create Sampler on pre-defined mode
        num_workers = self.exp_conf.num_loader_workers

        # train_sampler = self.dataloader.TrainSampler(
        #     data_conf=self.data_conf,
        #     dataset=self.trainset,
        #     batch_size=self.exp_conf.batch_size,
        #     sample_mode=self.exp_conf.sample_mode,
        # )
        '''
        Distributed Sampler
        '''
        train_sampler = self.dataloader.NewDistributedSampler(
                data_conf=self.data_conf,
                dataset=self.trainset,
                batch_size=self.exp_conf.batch_size,
                sample_mode=self.exp_conf.sample_mode,
            )

        train_loader = self.dataloader.create_data_loader(
            self.trainset,
            sampler=train_sampler,
            np_collate=False,
            length_batch=True,
            batch_size=self.exp_conf.batch_size // train_sampler.num_replicas,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
            max_squared_res=self.exp_conf.max_squared_res,
        )
        return train_loader

    def val_dataloader(self):
        num_workers = self.exp_conf.num_loader_workers
        valid_sampler = self.dataloader.NewDistributedSampler(
            data_conf=self.data_conf,
            dataset=self.valset,
            batch_size=self.data_conf.samples_per_eval_length,
            sample_mode=None,
            is_training=False
        )
        valid_loader = self.dataloader.create_data_loader(
            self.valset,
            sampler=valid_sampler,
            np_collate=False,
            length_batch=False,
            batch_size=self.data_conf.samples_per_eval_length // valid_sampler.num_replicas,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
        )
        return valid_loader

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