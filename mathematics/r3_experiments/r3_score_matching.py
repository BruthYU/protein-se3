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
from scipy.stats import wasserstein_distance_nd
from models.models import MLP
from lightning.data.framediff.r3_diffuser import R3Diffuser
torch.manual_seed(0)
from data.datasets import R3_Dataset, concat_np_features


# set up
os.environ['GEOMSTATS_BACKEND'] = 'pytorch'
_config.DEFAULT_DTYPE = torch.cuda.FloatTensor
savedir = "results/score_matching"
os.makedirs(savedir, exist_ok=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)



# Load toy dataset
dataset_name = "lorenz.npy"
data = np.load(f'data/{dataset_name}',allow_pickle=True)
print('size of toy dataset: ', len(data))
fig = plot_r3(data, title="Target Distribution")
plt.savefig(f"{savedir}/{dataset_name.split('.')[0]}.png", dpi=300)
plt.show()


# R3 Diffuser
r3_conf = OmegaConf.load('./config/r3_score_matching.yaml')
n_timestep = r3_conf.n_timestep
r3_diffuser = R3Diffuser(r3_conf)


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
    z = torch.randn_like(trans_0).to(device)
    # Apply noise
    trans_t = scheduler.sqrt_alphas_cumprod[t].view(-1, 1, 1) * trans_0 + \
              scheduler.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1) * z
    return trans_t, z


# DDPM Inference
def inference(model, trans_t, t):
    # rot_t rotation vector
    with torch.no_grad():
        t_tensor = (t / n_timestep).to(device)
        input = torch.cat([trans_t, t_tensor[:, None, None]], dim=-1)

        z_pred = model(input)
        w_z = (1. - scheduler.alphas[t]) / scheduler.sqrt_one_minus_alphas_cumprod[t]
        trans_mean = (1. / scheduler.sqrt_alphas[t]).view(-1, 1, 1) * (trans_t - w_z.view(-1, 1, 1) * z_pred)

        trans_z = torch.randn_like(trans_mean).to(device)
        trans_sigma = scheduler.sqrt_betas[t].view(-1, 1, 1)
        trans_next = trans_mean + trans_sigma * trans_z


    return trans_next


# Main Loop
def main_loop(model, optimizer, run_idx=0, num_epochs=150, display=True):
    losses = []
    w1ds = []

    final_traj = None
    for epoch in tqdm(range(num_epochs)):
        if (epoch % 10) == 0:
            n_test = testset.data.shape[0]
            traj= torch.randn((n_test, 1, 3)).to(device)
            steps = range(n_timestep, 0, -1)
            for t in steps:
                t = torch.tensor([t]).repeat(n_test)
                traj = inference(model, traj, t)
            final_traj = traj.squeeze().cpu().numpy()
            test_data = testset.data.squeeze()
            w_d1 = wasserstein_distance_nd(final_traj, test_data)
            w1ds.append(w_d1)

        if display and (epoch % 100)==0:
            plot_r3(final_traj, title='R3 DDPM')
            plt.savefig(os.path.join(savedir, f"dataset_{dataset_name.split('.')[0]}_run{run_idx}_epoch{epoch}.jpg"))
            plt.show()
            print('wassterstein-1 distance:', w_d1)

        # Train Model
        for _, data in enumerate(trainloader):
            trans_0 = torch.tensor(data)
            t = torch.randint(n_timestep, size=(trans_0.shape[0],)) + 1
            # Apply noise
            trans_t, z = trans_diffusion(trans_0.to(device), t.to(device))
            t_tensor = (t / n_timestep).to(device)
            input = torch.cat([trans_t, t_tensor[:,None,None]],dim=-1).to(device)
            z_pred = model(input)
            trans_loss = loss_fn(z_pred, z.to(device))
            trans_loss.backward()
            optimizer.step()
            losses.append(trans_loss.detach().cpu().numpy())
            optimizer.zero_grad()
    return model, np.array(losses), np.array(w1ds)





'''
Results for Multiple Runs
'''
w1ds_runs = []
losses_runs = []
num_runs = 3
for i in range(num_runs):
    print('doing run ', i)
    dim = 3  # network ouput is 3 dimensional (rot_vec matrix)
    model = MLP(dim=dim, time_varying=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    model, losses, w1ds, w2ds = main_loop(model, optimizer, run_idx=i, num_epochs=1000, display=True)

    w1ds_runs.append(w1ds)
    losses_runs.append(losses)

losses_runs = np.array(losses_runs)
w1ds_runs = np.array(w1ds_runs)

np.save(os.path.join(savedir, f"{dataset_name.split('.')[0]}_losses.npy"), losses_runs)
np.save(os.path.join(savedir, f"{dataset_name.split('.')[0]}_w1ds.npy",), w1ds_runs)
