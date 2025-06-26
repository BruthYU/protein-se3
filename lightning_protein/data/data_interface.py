import inspect
import importlib
import pytorch_lightning as pl
import random
import torch
from torch.utils.data import DataLoader
import logging
LOG = logging.getLogger(__name__)


class DInterface():
    def __init__(self, conf):
        # self.lightning_model
        self.conf = conf
        self.lightning_datamodule = self.init_lightning_datamodule(self.conf.method_name)
        self.datamodule = self.instancialize_lightning_model(self.lightning_datamodule, self.conf)

    def init_lightning_datamodule(self, name):
        return getattr(importlib.import_module(f'data.{name}.lightning_datamodule'), f'{name}_Lightning_Datamodule')

    def instancialize_lightning_model(self, datamodule, conf):
        return datamodule(conf)