import os
import matplotlib.pyplot as plt
import torch
from geomstats._backend import _backend_config as _config
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from utils.plotting import plot_r3
from data.datasets import R3Dataset
from utils.ddpm_utils import r3_ddpm_scheduler
from scipy.stats import wasserstein_distance_nd
from models.models import MLP
torch.manual_seed(0)

os.environ['GEOMSTATS_BACKEND'] = 'pytorch'
_config.DEFAULT_DTYPE = torch.cuda.FloatTensor
savedir = "results/ddpm"
os.makedirs(savedir, exist_ok=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
r3_conf = OmegaConf.load('./config/r3_ddpm.yaml')
n_timestep = r3_conf.diffusion.n_timestep

# Load toy dataset
dataset_name = "lorenz.npy"
data = np.load(f'data/{dataset_name}',allow_pickle=True)
print('size of toy dataset: ', len(data))
fig = plot_r3(data, title="Target Distribution")
plt.savefig(f"{savedir}/{dataset_name.split('.')[0]}.png", dpi=300)
plt.show()


# DDPM scheduler
scheduler = r3_ddpm_scheduler(r3_conf)


# Load Dataset

trainset = R3Dataset(split="train", name=dataset_name)
trainloader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=0)
testset = R3Dataset(split="test", name=dataset_name)
testloader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=0)




def loss_fn(z_pred, z):
    # sample noise
    mse = torch.nn.MSELoss(reduction='sum')
    loss = mse(z_pred, z)
    return loss

# Add Noise
def trans_diffusion(trans_0, t):
    z = torch.randn_like(trans_0)
    # Apply noise
    trans_t = scheduler.sqrt_alphas_cumprod[t].view(-1, 1, 1) + \
              scheduler.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1) * z
    return trans_t


# DDPM Inference
def inference(model, trans_t, t):
    # rot_t rotation vector
    with torch.no_grad():
        t_tensor = t / n_timestep
        input = torch.cat([trans_t, t_tensor[:, None, None]], dim=-1)
        # rotvec_next, rotmatrix_next = model(input)
        z_pred = model(input)
        w_z = (1. - scheduler.alphas[t]) / scheduler.sqrt_one_minus_alphas_cumprod[t]
        trans_next = (1. / scheduler.sqrt_alphas[t]).view(-1, 1, 1) * (trans_t - w_z.view(-1, 1, 1) * z_pred)
    return trans_next


# Main Loop
def main_loop(model, optimizer, run_idx=0, num_epochs=150, display=True):
    losses = []
    w1ds = []

    final_traj = None
    for epoch in tqdm(range(num_epochs)):
        if (epoch % 10) == 0:
            n_test = testset.data.shape[0]
            traj= torch.randn((n_test, 1, 3))
            steps = range(n_timestep, 0, -1)
            for t in steps:
                t = torch.tensor([t]).to(device).repeat(n_test)
                traj = inference(model, traj, t)
            final_traj = traj.sqeenze().cpu().numpy()
            test_data = testset.data.squeeze()
            w_d1 = wasserstein_distance_nd(final_traj, test_data)
            w1ds.append(w_d1)

        if display and (epoch % 100)==0:
            plot_r3(final_traj, adjust=True, title='R3 DDPM')
            plt.savefig(os.path.join(savedir, f"dataset_{dataset_name.split('.')[0]}_run{run_idx}_epoch{epoch}.jpg"))
            plt.show()
            print('wassterstein-1 distance:', w_d1)

        # Train Model
        for _, data in enumerate(trainloader):
            trans_0 = torch.tensor(data).to(device)

            t = torch.randint(n_timestep, size=(rotvec_0.shape[0],)).to(device) + 1
            rotvec_t = rot_diffusion(rotvec_0.cpu(), t.cpu()).to(device)
            t_tensor = t / n_timestep
            input = torch.cat([rotvec_t, t_tensor[:,None,None]],dim=-1)
            rotvec_pred = model(input)
            # rotvec_pred, rotmatrix_pred = model(input)
            # rot_loss = rotation_matrix_cosine_loss(rotmatrix_pred, rotmatrix_0).sum()
            rot_loss = loss_fn(rotvec_pred, rotvec_0)
            rot_loss.backward()
            optimizer.step()
            losses.append(rot_loss.detach().cpu().numpy())
            optimizer.zero_grad()
    return model, np.array(losses), np.array(w1ds), np.array(w2ds)



# dim = 3  # network ouput is 3 dimensional (rot_vec matrix)
# model = UMLP(dim=dim, time_varying=True).to(device).double()
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
# model, losses, w1ds, w2ds = main_loop(model, optimizer, num_epochs=1000, display=True)

'''
Results for Multiple Runs
'''
w1ds_runs = []
w2ds_runs = []
losses_runs = []
num_runs = 3
for i in range(num_runs):
    print('doing run ', i)
    dim = 3  # network ouput is 3 dimensional (rot_vec matrix)
    model = UMLP(dim=dim, time_varying=True).double().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    model, losses, w1ds, w2ds = main_loop(model, optimizer, run_idx=i, num_epochs=1000, display=True)

    w1ds_runs.append(w1ds)
    w2ds_runs.append(w2ds)
    losses_runs.append(losses)

losses_runs = np.array(losses_runs)
w1ds_runs = np.array(w1ds_runs)
w2ds_runs = np.array(w2ds_runs)

np.save(os.path.join(savedir, f"{dataset_name.split('.')[0]}_losses.npy"), losses_runs)
np.save(os.path.join(savedir, f"{dataset_name.split('.')[0]}_w1ds.npy",), w1ds_runs)
np.save(os.path.join(savedir, f"{dataset_name.split('.')[0]}_w2ds.npy"), w2ds_runs)