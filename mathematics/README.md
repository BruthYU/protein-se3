# Toolkit Guide
[![arXiv](https://img.shields.io/badge/arXiv-2507.20243-B31B1B.svg)](https://arxiv.org/abs/2507.20243)
![Static Badge](https://img.shields.io/badge/PRs-Welcom-green)
![Star](https://img.shields.io/github/stars/BruthYU/protein-se3?style=social&label=Star)

The underlying mathematical principles of $\texttt{SE(3)}$-based protein design methods are often highly non-trivial and fragmented. 
We build this toolkit with intuitive demos, which could serve as the entry point for junior researchers to understand the theories.
**Detailed formulations and experimental results of each method can be found in our paper**.

### Tips
- To use the evaluation metric `wasserstein_distance_nd` for $\mathbb{R}^3$ experiments, `scipy` is required to installed in `Python>=3.10`. 


### $\mathbb{R}^3$ Experiments
- We constructed two target distributions `r3_experiments/data/lorenz.npy` and `r3_experiments/data/sine.npy`, which store two sets of spatial point distributions in the form of 3D coordinates.
The experiments are built upon simple MLP layers training on the synthetic data (3D coordinates), 
enabling systematic study on how various generative models (DDPM, Score Matching, and Flow Matching) align distributions in the $\mathbb{R}^3$ space, which is measured by the 1st-order Wasserstein distance
```shell
# To observe the aliment process of DDPM in r3, simply run:
python r3_ddpm.py
```



| ![](../document/r3_crop/lorenz.png) | ![](../document/r3_crop/lorenz/ddpm.jpg) | ![](../document/r3_crop/lorenz/score_matching.jpg) | ![](../document/r3_crop/lorenz/flow_matching.jpg) |
|-------------------------------------|------------------------------------------|---------------------------------------------------|---------------------------------------------------|

| ![](../document/r3_crop/sine.png) | ![](../document/r3_crop/sine/ddpm.jpg) | ![](../document/r3_crop/sine/score_matching.jpg) | ![](../document/r3_crop/sine/flow_matching.jpg) |
|-----------------------------------|----------------------------------------|--------------------------------------------------|-------------------------------------------------|


| ![](../document/lorenz_w1ds.png) | ![](../document/sine_w1ds.png)|
|----------------------------------|-------------------------|

### $\texttt{SO(3)}$ Experiments
We constructed two target distributions `so3_experiments/data/bunny_group.npy` and `r3_experiments/data/spiral_group.npy`, which store two sets of rotation matrices in the form of **Euler Angle Representation**.
The experiments are built upon simple MLP layers training on the synthetic data (rotation matrices), 
enabling systematic study on how various generative models (DDPM, Score Matching, and Flow Matching) align distributions in the $\texttt{SO(3)}$ space, which is measured by the 1st-order Wasserstein distance
```shell
# To observe the aliment process of DDPM in so3, simply run:
python so3_ddpm.py
```
| ![](../document/so3_crop/bunny_group.png) | ![](../document/so3_crop/bunny/ddpm.jpg) | ![](../document/so3_crop/bunny/score_matching.jpg) | ![](../document/so3_crop/bunny/flow_matching.jpg) |
|-------------------------------------------|------------------------------------------|---------------------------------------------------|---------------------------------------------------|

| ![](../document/so3_crop/spiral_group.png) | ![](../document/so3_crop/spiral/ddpm.jpg) | ![](../document/rso3_crop/spiral/score_matching.jpg) | ![](../document/so3_crop/spiral/flow_matching.jpg) |
|--------------------------------------------|-------------------------------------------|--------------------------------------------------|-------------------------------------------------|


| ![](../document/bunny_group_w1ds.png) | ![](../document/spiral_group_w1ds.png) |
|---------------------------------------|----------------------------------------|

