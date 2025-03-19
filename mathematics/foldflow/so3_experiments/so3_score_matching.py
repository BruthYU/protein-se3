import os
import numpy as np
import torch
from einops import rearrange
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import warnings
import tree
warnings.filterwarnings('ignore')


from scipy.spatial.transform import Rotation
from geomstats.geometry.special_orthogonal import SpecialOrthogonal

from utils.plotting import plot_so3
from utils.optimal_transport import so3_wasserstein as wasserstein
from lightning.data.foldflow.so3_helpers import norm_SO3, expmap
from lightning.data.foldflow.so3_condflowmatcher import SO3ConditionalFlowMatcher
from mathematics.foldflow.so3_experiments.models.models import MLP
from torch.utils.data import DataLoader
from data.datasets import SpecialOrthogonalGroup
from data.datasets import concat_np_features
from geomstats._backend import _backend_config as _config
from lightning.data.framediff.so3_diffuser import SO3Diffuser


_config.DEFAULT_DTYPE = torch.cuda.FloatTensor
savedir = "models/so3_synthetic"
os.makedirs(savedir, exist_ok=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)



'''
Load toy dataset
'''
data = np.load('data/bunny_group.npy',allow_pickle=True)
print('size of toy dataset: ', len(data))

fig = plot_so3(data, adjust=True)
plt.savefig('figs/so3_synthetic_data.png', dpi=300)
plt.show()


'''
Load Dataset
'''
dataset_name = "orthogonal_group.npy"
trainset = SpecialOrthogonalGroup(split="train", dataset_name=dataset_name)
trainloader = DataLoader(
    trainset, batch_size=1024, shuffle=True, num_workers=0, collate_fn=lambda x: concat_np_features(x, add_batch_dim=True)
)

valset = SpecialOrthogonalGroup(split="valid",dataset_name=dataset_name)
valloader = DataLoader(
    valset, batch_size=256, shuffle=False, num_workers=0, collate_fn=lambda x: concat_np_features(x, add_batch_dim=True)
)

testset = SpecialOrthogonalGroup(split="test", dataset_name=dataset_name)
testloader = DataLoader(
    testset, batch_size=256, shuffle=False, num_workers=0, collate_fn=lambda x: concat_np_features(x, add_batch_dim=True)
)

'''
SO3 Diffuser
'''
from omegaconf import OmegaConf
so3_conf = OmegaConf.load('./config/score_matching.yaml')
so3_diffuser = SO3Diffuser(so3_conf.so3)


'''
Model
'''
dim = 9 # network ouput is 9 dimensional (3x3 matrix)
model = MLP(dim=dim, time_varying=True).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

'''
Training Loop
'''

def diff_rot(data, state=0):
    diff_rot_feats = {}
    rot_t_list = []
    rot_score_list = []
    rot_score_scaling_list = []
    rng = np.random.default_rng(state)
    t_list = [rng.uniform(so3_conf.so3.min_t, 1.0) for _ in range(len(data))]
    for rot_0, t in zip(data, t_list):
        rot_t, rot_score = so3_diffuser.forward_marginal(
            rot_0, t)
        rot_score_scaling = so3_diffuser.score_scaling(t)
        rot_t_list.append(rot_t)
        rot_score_list.append(rot_score)
        rot_score_scaling_list.append(rot_score_scaling)
    diff_rot_feats['t'] = np.stack(t_list)
    diff_rot_feats['rot_t'] = np.stack(rot_t_list)
    diff_rot_feats['rot_score'] = np.stack(rot_score_list)
    diff_rot_feats['rot_score_scaling'] = np.stack(rot_score_scaling_list)
    final_feats = tree.map_structure(
        lambda x: x if torch.is_tensor(x) else torch.tensor(x), diff_rot_feats)
    return final_feats


def main_loop(model, optimizer, num_epochs=150, display=True):
    losses = []
    w1ds = []
    w2ds = []
    global_step = 0
    for epoch in range(num_epochs):

        for _, batch in enumerate(trainloader):
            data = batch['rotationMatrix']
            diff_rot_feats = diff_rot(data)
            optimizer.zero_grad()

main_loop(model, optimizer, num_epochs=80)