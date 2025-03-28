import os

import matplotlib.pyplot as plt
from geomstats._backend import _backend_config as _config
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.datasets import DDPM_Dataset
from mathematics.foldflow.so3_experiments.models.models import MLP
from utils.ddpm_utils import *
from utils.optimal_transport import so3_wasserstein as wasserstein
from utils.plotting import plot_so3

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
Rotation_DDPM = RotationTransition(num_steps=so3_conf.diffusion.n_timestep)


# Load Dataset
dataset_name = "bunny_group.npy"
trainset = DDPM_Dataset(split="train", name=dataset_name)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=0)
testset = DDPM_Dataset(split="test", name=dataset_name)
testloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=0)


# loss_fn
def rotation_matrix_cosine_loss(R_pred, R_true):
    """
    Args:
        R_pred: (*, 3, 3).
        R_true: (*, 3, 3).
    Returns:
        Per-matrix losses, (*, ).
    """
    size = list(R_pred.shape[:-2])
    ncol = R_pred.numel() // 3

    RT_pred = R_pred.transpose(-2, -1).reshape(ncol, 3) # (ncol, 3)
    RT_true = R_true.transpose(-2, -1).reshape(ncol, 3) # (ncol, 3)

    ones = torch.ones([ncol, ], dtype=torch.long, device=R_pred.device)
    loss = F.cosine_embedding_loss(RT_pred, RT_true, ones, reduction='none')  # (ncol*3, )
    loss = loss.reshape(size + [3]).sum(dim=-1)    # (*, )
    return loss

def loss_fn(rot_pred, rot_0):
    # sample noise
    mse = torch.nn.MSELoss(reduction='sum')
    loss = mse(rot_pred, rot_0)
    return loss

# Add Noise
def rot_diffusion(rot_0, t):
    # rot_0 [N,3]
    rot_t, _ = Rotation_DDPM.add_noise(rot_0[:, None, :], t) # []
    return rot_t


# DDPM Inference
def inference(model, rot_t, t):
    # rot_t rotation vector
    with torch.no_grad():
        input = torch.cat([rot_t, t[:, None]], dim=-1)
        rot_next = model(input)
        # Compute posterior

        rot_next = Rotation_DDPM.denoise(rot_next[:, None, :], t.cpu())
    return rot_next.squeeze().to(device)


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
            steps = range(so3_conf.diffusion.n_timestep, 0, -1)
            for t in steps:
                t = torch.tensor([t]).to(device).repeat(n_test)
                traj = inference(model, traj, t)
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
            t = torch.randint(so3_conf.diffusion.n_timestep, size=(rot_0.shape[0],)).to(device) + 1
            rot_t = rot_diffusion(rot_0.cpu(), t.cpu()).to(device)
            input = torch.cat([rot_t.squeeze(), t[:,None]],dim=-1)
            rot_pred = model(input)
            rot_loss = loss_fn(rot_pred, rot_0)
            rot_loss.backward()
            optimizer.step()
            losses.append(rot_loss.detach().cpu().numpy())
            optimizer.zero_grad()
    return model, np.array(losses), np.array(w1ds), np.array(w2ds)



dim = 3  # network ouput is 3 dimensional (rot_vec matrix)
model = MLP(dim=dim, time_varying=True).to(device).double()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
model, losses, w1ds, w2ds = main_loop(model, optimizer, num_epochs=1000, display=True)


