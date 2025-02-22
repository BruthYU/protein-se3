from omegaconf import DictConfig
import time
import hydra
import logging
from sampler import SInterface


LOG = logging.getLogger(__name__)
@hydra.main(version_base=None, config_path="config", config_name="inference")
def run(conf: DictConfig) -> None:
    Sampler = SInterface(conf)

    # Sampler Instance: Sampler.sampler
    LOG.info('Starting inference')
    start_time = time.time()
    Sampler.sampler.run_sampling()
    elapsed_time = time.time() - start_time
    LOG.info(f'Finished in {elapsed_time:.2f}s')


if __name__ == '__main__':
    run()