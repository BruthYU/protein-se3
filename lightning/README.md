# Lightning Workspace
Main workspace of protein-se3, which contains the Lightning⚡ implementations of collective
**Protein Structure Generative Models**. 

## Folders and Files
<details>

1. **config**: protein-se3 manages complex configuration with the [hydra](https://github.com/facebookresearch/hydra) framework. 
This folder contains default settings of integrated methods. Specifically, `train.yaml` and `inference.yaml` select method for training or inference
by setting the value `default`, corresponding configurations are loaded from the folder `config/method`. For example, if we want to run training of **FoldFlow**, we can set the `train.yaml` as
    ```yaml
    # config/train.yaml
    defaults:
      - method: foldflow
    ```
   and simply run `main_train.py` (similar to the inference).
2. **data**: With loaded `lmdb` cache, methods further extract features (e.g. frame with t-step diffusion) to determine the dataloader for training and inference.
In every folder for each method, a `lightning_datamodule.py` are implemented to align the interface `ligtning/data/data_interface.py`. Note that restrict datamodule class names
are required (`{}_Lightning_Datamodule`).
   ```python
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
   ```
3. **model**: In line with the deep learning framework [Pytorch Lightning](https://lightning.ai/docs/pytorch/stable/), both model architecture and training details (e.g. training step and loss function) are placed in this folder.
In every folder for each method, a `lightning_model.py` are implemented to align the interface `ligtning/model/model_interface.py`. Note that restrict model class names
are required (`{}_Lightning_Model`).
      ```python
   class MInterface():
    def __init__(self, conf):
        # self.lightning_model
        self.conf = conf
        self.lightning_model = self.init_lightning_model(self.conf.method_name)
        self.model = self.instancialize_lightning_model(self.lightning_model, self.conf)

    def init_lightning_model(self, name):
        return getattr(importlib.import_module(f'model.{name}.lightning_model'), f'{name}_Lightning_Model')

    def instancialize_lightning_model(self, model, conf):
        return model(conf)
   ```
4. **sampler**: For convenient usage of pre-trained model, we develop this folder supporting checkpoint loading and protein sampling.
In every folder for each method, a `sampler_module.py` are implemented to align the interface `ligtning/sampler/sampler_interface.py`. Note that restrict sampler class names
are required (`{}_Sampler`).
   ```Python
   class SInterface():
       def __init__(self, conf):
           # self.lightning_model
           self.conf = conf
           self.sampler_module = self.init_sampler_module(self.conf.method_name)
           self.sampler = self.instancialize_lightning_model(self.sampler_module, self.conf)
   
       def init_sampler_module(self, name):
           return getattr(importlib.import_module(f'sampler.{name}.sampler_module'), f'{name}_Sampler')
   
       def instancialize_lightning_model(self, sampler, conf):
           return sampler(conf)
   ```

</details>

## Tutorial

### Prepare Datasets
Following this [README.md](../preprocess/README.md) to preprocess and prepare datasets. 
### Training
- **Step 1: Configure Generative Model.** protein-se3 currently supports *FrameDiff*, *FoldFLow*, *Genie2*,
*FrameFlow* and *RFDiffusion*. You can choose any protein generative method (e.g. *Genie2*) by setting `config/train.yaml` as:
   ```yaml
   defaults:
     - method: genie2
   ```
  Detailed configurations of each specific method are placed in folder `config/method`, You can modify it as needed.
- **Step 2: Run Training.** `lightning/main_train.py` is the main script for training. As configurations are managed with [hydra](https://github.com/facebookresearch/hydra) framework, you can simply run training with
    ```shell
    CUDA_VISIBLE_DEVICES=0,1 python main_train.py
    ```
  In `lightning/main_train.py`, you can configure things about training, such as whether to use [WandB](https://wandb.ai/site/) to supervise training,
or which visible GPUs are used. With our careful implementation based on Pytorch Lightning⚡, distributed training and evaluation will run automatically.
- **Step 3: Training Process.** We use [hydra](https://github.com/facebookresearch/hydra) to create a directory for each run to store
outputs during training (such as checkpoints and evaluation results):
    ```yaml
    # config/traing.yaml
    hydra:
      run:
        dir: ./hydra_train/${now:%Y-%m-%d_%H-%M-%S}_${method_name}
      job:
        chdir: True
    ```
  The created output directories during training will be like:
    ```
    ├── hydra_train
    │         ├── 2024-11-26_20-08-24_genie2
    │         ├── 2024-11-26_20-09-11_genie2
    ```

---
### Inference
- **Create Resource Directory:** Before inference, you need to create a folder `lightning/resource` to place resources for inference (such as pre-trained checkpoints 
and task files for motif scaffolding)
    ```text
    lightning/resource
     └── frameflow
        ├── targets
        ├── benchmark.csv 
        └── last.ckpt
    └── genie2
        ├── design25
        └── last.ckpt
    ```

- **Configure Generative Model:** You should first choose a pre-trained protein generative (e.g. *FoldFlow*) model in `config/inference.yaml` as:
   ```yaml
   defaults:
     - method: foldflow
   ```
  Import configurations in `config/method/foldflow.yaml` to perform inference with *FoldFLow*:
   ```yaml
   inference:
     weights_path: ${hydra:runtime.cwd}/src/foldflow/last.ckpt # Path to model weights.
     output_dir: ./foldflow_outputs # Inference Output directory 
   ```
  The created folder for inference output will be like: 
    ```
    ├── hydra_inference
    │         ├── 2024-11-26_20-08-24_foldflow
    │         │   ├── foldflow_outputs
    ```
- **Task Type:** The types of tasks supported by different methods are listed as follows:
    
    | **Task**           | FrameDiff  | FoldFlow | Genie2 | FrameFlow | RFDiffusion |
    |:-------------------|:----------:|:--------:|:------:|:---------:|:-----------:|
    | Unconditional Gen. |    ✅       |     ✅   |   ✅    |     ✅     |      ✅      |
    | Motif Scaffolding  |            |          |   ✅    |     ✅     |      ✅     |

    For unconditional generatoin, after defining your task configurations (such as protein lengths), you can simply run:
    ```shell
    python main_inference.py
    ```
    For methods supporting motif scaffolding (e.g. *Genie2*), you should first determine the task type in
`conifg/mehtod/genie2.yaml` as:
    ```yaml
   inference:
      task_type: scaffold # scaffold 
    ```
   Please refer to each original repo to see the 
- *Genie2*: [Format of Motif Scaffolding Problem Definition File](https://github.com/aqlaboratory/genie2/blob/9a954578f7b5a39552545eebc6d4794447794c87/README.md?plain=1#L135).
- *FrameFlow*: [README.md: Motif-scaffolding](https://github.com/microsoft/protein-frame-flow/blob/f50d8dbbdae827be291e9f73d732b61b195f8816/README.md?plain=1#L139)  

