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
dataset_name = "bunny_group.npy"
trainset = SpecialOrthogonalGroup(split="train", dataset_name=dataset_name)
trainloader = DataLoader(
    trainset, batch_size=256, shuffle=True, num_workers=0, collate_fn=lambda x: concat_np_features(x, add_batch_dim=True)
)

valset = SpecialOrthogonalGroup(split="valid",dataset_name=dataset_name)
valloader = DataLoader(
    valset, batch_size=128, shuffle=False, num_workers=0, collate_fn=lambda x: concat_np_features(x, add_batch_dim=True)
)

testset = SpecialOrthogonalGroup(split="test", dataset_name=dataset_name)
testloader = DataLoader(
    testset, batch_size=128, shuffle=False, num_workers=0, collate_fn=lambda x: concat_np_features(x, add_batch_dim=True)
)

'''
SO3 Diffuser
'''
from omegaconf import OmegaConf
so3_conf = OmegaConf.load('./config/score_matching.yaml')
so3_diffuser = SO3Diffuser(so3_conf.so3)



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
        rot_0 = Rotation.from_matrix(rot_0).as_rotvec()
        rot_t, rot_score = so3_diffuser.forward_marginal(
            rot_0[None], t)
        rot_score_scaling = so3_diffuser.score_scaling(t)
        rot_t_list.append(rot_t)
        rot_score_list.append(rot_score)
        rot_score_scaling_list.append(rot_score_scaling)
    diff_rot_feats['t'] = np.stack(t_list)
    diff_rot_feats['rot_t'] = np.concatenate(rot_t_list)
    diff_rot_feats['rot_score'] = np.concatenate(rot_score_list)
    diff_rot_feats['rot_score_scaling'] = np.stack(rot_score_scaling_list)
    final_feats = tree.map_structure(
        lambda x: x if torch.is_tensor(x) else torch.tensor(x), diff_rot_feats)
    return final_feats




'''
SDE Inference
'''
def inference(model, rot_t, t, dt, noise_scale=1.0):
    with torch.no_grad():

        input = torch.cat([rot_t, t[:,None]],dim=-1)
        rot_score = model(input)

        rot_t = rot_t.cpu().numpy()
        rot_score = rot_score.cpu().numpy()

        rot_t_1 = so3_diffuser.reverse(
            rot_t=rot_t,
            score_t=rot_score,
            t=t[0].item(),
            dt=dt.item(),
            noise_scale=noise_scale,
        )
    return torch.tensor(rot_t_1).to(device)


'''
Main Loop
'''
def main_loop(model, optimizer, num_epochs=150, display=True):
    losses = []
    w1ds = []
    w2ds = []


    global_step = 0
    for epoch in tqdm(range(num_epochs)):


        if (epoch % 100) == 0:
            n_test = testset.data.shape[0]
            traj = torch.tensor(Rotation.random(n_test).as_rotvec()).to(device)
            for t in torch.linspace(so3_conf.so3.min_t, 1, 100):
                t = torch.tensor([t]).to(device).repeat(n_test).requires_grad_(True)
                dt = torch.tensor([1 / 200]).to(device)
                traj = inference(model, traj, t, dt)
            final_traj = traj
            final_traj = Rotation.from_rotvec(final_traj.cpu().numpy()).as_matrix()

            w_d1 = wasserstein(torch.tensor(testset.data).to(device).double().detach(),
                               torch.tensor(final_traj).to(device).double().detach(), power=1)
            w_d2 = wasserstein(torch.tensor(testset.data).to(device).double().detach(),
                               torch.tensor(final_traj).to(device).double().detach(), power=2)
            w1ds.append(w_d1)
            w2ds.append(w_d2)


            if display:
                plot_so3(final_traj)
                plt.show()
                print('wassterstein-1 distance:', w_d1)
                print('wassterstein-2 distance:', w_d2)

        for _, batch in enumerate(trainloader):
            data = batch['rotationMatrix']
            diff_rot_feats = diff_rot(data)
            for key in diff_rot_feats.keys():
                diff_rot_feats[key] = diff_rot_feats[key].to(device)
            rot_t = diff_rot_feats['rot_t']
            input = torch.cat([rot_t, diff_rot_feats['t'][:,None]],dim=-1)
            pred_rot_score = model(input)

            loss = torch.nn.MSELoss()
            rot_mse = (diff_rot_feats['rot_score']- pred_rot_score) ** 2
            rot_loss = torch.sum(
                rot_mse / diff_rot_feats['rot_score_scaling'][:, None] ** 2)
            rot_loss.backward()

            optimizer.step()
            losses.append(rot_loss.detach().cpu().numpy())


            optimizer.zero_grad()
    return model, np.array(losses), np.array(w1ds), np.array(w2ds)







'''
Results for Multiple Runs
'''
w1ds_runs = []
w2ds_runs = []
losses_runs = []
num_runs = 5
for i in range(num_runs):
    print('doing run ', i)
    dim = 3  # network ouput is 3 dimensional (rot_vec matrix)
    model = MLP(dim=dim, time_varying=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5, foreach=False)
    model, losses, w1ds, w2ds = main_loop(model, optimizer, num_epochs=1000, display=True)

    w1ds_runs.append(w1ds)
    w2ds_runs.append(w2ds)
    losses_runs.append(losses)

losses_runs = np.array(losses_runs)
w1ds_runs = np.array(w1ds_runs)
w2ds_runs = np.array(w2ds_runs)


'''
Plot Results
'''
losses_runs = np.array(losses_runs)
w1ds_runs = np.array(w1ds_runs)
w2ds_runs = np.array(w2ds_runs)
# mean of w1s
w1s_mean = np.mean(w1ds_runs, axis=0)
w1s_std = np.std(w1ds_runs, axis=0)
# mean of w2s
w2s_mean = np.mean(w2ds_runs, axis=0)
w2s_std = np.std(w2ds_runs, axis=0)

print('w1s_mean', w1s_mean[-1])
print('w1s_std', w1s_std[-1])
print('w2s_mean', w2s_mean[-1])
print('w2s_std', w2s_std[-1])

import seaborn as sns
sns.set_theme(style="darkgrid")

fig, ax = plt.subplots(1, 2, figsize=(18, 5))

x = np.arange(0, 80, 10)

ax[0].plot(x, w1s_mean)
ax[0].fill_between(x, w1s_mean - w1s_std, w1s_mean + w1s_std, color='C0', alpha=0.4)
ax[0].set_xlabel('epoch', fontsize=14)
ax[0].set_ylabel('1-Wasserstein distance', fontsize=14)

ax[1].plot(x, w2s_mean)
ax[1].fill_between(x, w2s_mean - w2s_std, w2s_mean + w2s_std, color='C0', alpha=0.4)
ax[1].set_xlabel('epoch', fontsize=14)
ax[1].set_ylabel('2-Wasserstein-2 distance', fontsize=14)
plt.show()


'''
Plot losses
'''
print(losses_runs.shape)
plt.plot(losses_runs[1])
plt.xlabel('iteration')
plt.ylabel('loss')
plt.show()


'''inference on the full dataset for visualization'''

plt.style.use('default')
n_test = data.shape[0]
traj = torch.tensor(Rotation.random(n_test).as_matrix()).to(device).reshape(-1, 9)
for t in torch.linspace(0, 1, 200):
    t = torch.tensor([t]).to(device).repeat(n_test)
    dt = torch.tensor([1/200]).to(device)
    traj = inference(model, traj, t, dt)
final_traj = rearrange(traj, 'b (c d) -> b c d', c=3, d=3)
fig = plot_so3(final_traj)
plt.show()