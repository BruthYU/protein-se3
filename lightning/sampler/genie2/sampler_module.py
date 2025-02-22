from tqdm import tqdm

from lightning.sampler.genie2.unconditional_runner import UnconditionalRunner
from lightning.sampler.genie2.scaffold_runner import ScaffoldRunner

from lightning.sampler.genie2.multiprocessor import MultiProcessor

from omegaconf import DictConfig, OmegaConf
import torch
import logging

class genie2_Sampler:
    def __init__(self, conf: DictConfig):
        self.log = logging.getLogger(__name__)

        if conf.inference.task_type == "unconditional":
            self.runner = UnconditionalRunner()
            self.infer_conf = conf.inference.unconditional
        elif conf.inference.task_type == "scaffold":
            self.runner = ScaffoldRunner()
            self.infer_conf = conf.inference.scaffold

        self.output_dir = self.infer_conf.output_dir
        self.num_devices = torch.cuda.device_count()
        self.log.info(f'Inference on {self.num_devices} GPUs.')




    def run_sampling(self):
        self.runner.run(self.infer_conf, self.num_devices)







