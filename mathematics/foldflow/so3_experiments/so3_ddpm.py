import torch
from geomstats._backend import _backend_config as _config
import os
import math
import matplotlib.pyplot as plt
import numpy as np
from utils.plotting import plot_so3
from utils.so3_ddpm_scheduler import SO3_DDPM_Scheduler
from data.datasets import DDPM_Dataset
from torch.utils.data import DataLoader
import tqdm
from scipy.spatial.transform import Rotation
from omegaconf import OmegaConf
from utils.optimal_transport import so3_wasserstein as wasserstein
from mathematics.foldflow.so3_experiments.models.models import MLP
so3_conf = OmegaConf.load('./config/ddpm.yaml')
os.environ['GEOMSTATS_BACKEND'] = 'pytorch'
_config.DEFAULT_DTYPE = torch.cuda.FloatTensor
savedir = "models/so3_synthetic"
os.makedirs(savedir, exist_ok=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# Load toy dataset
data = np.load('data/bunny_group.npy',allow_pickle=True)
print('size of toy dataset: ', len(data))
fig = plot_so3(data, adjust=True)
plt.savefig('figs/so3_synthetic_data.png', dpi=300)
plt.show()


# SO3 DDPM Scheduler
so3_scheduler = SO3_DDPM_Scheduler(so3_conf.diffusion)


# Load Dataset
dataset_name = "bunny_group.npy"
trainset = DDPM_Dataset(split="train", name=dataset_name)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=0)
testset = DDPM_Dataset(split="test", name=dataset_name)
testloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=0)


# loss_fn
def loss_fn(z_pred):
    # sample noise
    z = torch.randn_like(z_pred)
    loss = torch.nn.MSELoss(z_pred, z, reduction='sum')
    return loss


# Add Noise
def rot_diffusion(scheduler, rot_0, t):
    z = torch.randn_like(rot_0)
    rot_t = scheduler.sqrt_alphas_cumprod[t].view(-1, 1, 1) * rot_0 + \
              scheduler.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1) * z
    return rot_t


# DDPM Inference
def inference(model, rot_t, t, scheduler):
    # rot_t rotation vector
    with torch.no_grad():
        input = torch.cat([rot_t, t[:,None]],dim=-1)
        z_pred = model(input)
        # Compute posterior
        w_z = (1. - scheduler.alphas[t]) / scheduler.sqrt_one_minus_alphas_cumprod[t]
        rot_t_1 = (1. / scheduler.sqrt_alphas[t]).view(-1, 1, 1) * (rot_t - w_z.view(-1, 1, 1) * z_pred)
    return torch.tensor(rot_t_1).to(device)


# Main Loop
def main_loop(model, optimizer, num_epochs=150, display=True):
    losses = []
    w1ds = []
    w2ds = []

    global_step = 0
    for epoch in tqdm(range(num_epochs)):
        if (epoch % 100) == 0:
            n_test = testset.data.shape[0]
            traj = torch.tensor(Rotation.random(n_test).as_rotvec()).to(device)
            steps = reversed(np.arange(1, so3_conf.diffsuion.n_timestep + 1))
            for t in steps:
                t = torch.tensor([t]).to(device).repeat(n_test).requires_grad_(True)
                traj = inference(model, traj, t, so3_scheduler)
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

        for _, data in enumerate(trainloader):
            rot_0 = torch.tensor(data).to(device)
            t = torch.randint(so3_conf.diffusion.n_timestep, size=(rot_0.shape[0])).to(device) + 1
            rot_t = rot_diffusion(so3_scheduler, rot_0, t)
            input = torch.cat([rot_t, t[:,None]],dim=-1)
            z_pred = model(input)
            rot_loss = loss_fn(z_pred)
            rot_loss.backward()
            optimizer.step()
            losses.append(rot_loss.detach().cpu().numpy())
            optimizer.zero_grad()
    return model, np.array(losses), np.array(w1ds), np.array(w2ds)



dim = 3  # network ouput is 3 dimensional (rot_vec matrix)
model = MLP(dim=dim, time_varying=True).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
model, losses, w1ds, w2ds = main_loop(model, optimizer, num_epochs=1000, display=True)


