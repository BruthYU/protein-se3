import os
import matplotlib.pyplot as plt
import torch
from geomstats._backend import _backend_config as _config
from omegaconf import OmegaConf

from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from utils.plotting import plot_r3

from scipy.stats import wasserstein_distance_nd
from models.models import MLP
from lightning.data.framediff.r3_diffuser import R3Diffuser
torch.manual_seed(0)
from data.datasets import R3_SDE_Dataset, concat_np_features


# set up
os.environ['GEOMSTATS_BACKEND'] = 'pytorch'
_config.DEFAULT_DTYPE = torch.cuda.FloatTensor
savedir = "results/score_matching"
os.makedirs(savedir, exist_ok=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)



# Load toy dataset
dataset_name = "sine.npy"
data = np.load(f'data/{dataset_name}',allow_pickle=True)
print('size of toy dataset: ', len(data))
fig = plot_r3(data, title=r"Target $R^3$ Distribution A")
plt.savefig(f"{savedir}/{dataset_name.split('.')[0]}.png", dpi=300)
plt.show()


# R3 Diffuser
r3_conf = OmegaConf.load('./config/r3_score_matching.yaml')
r3_diffuser = R3Diffuser(r3_conf)


# Load Dataset
collate_fn = lambda x: concat_np_features(x, add_batch_dim=True)

trainset = R3_SDE_Dataset(split="train", name=dataset_name, r3_diffuser=r3_diffuser)
trainloader = DataLoader(
    trainset, batch_size=64, shuffle=True, num_workers=0, collate_fn=collate_fn
)


testset = R3_SDE_Dataset(split="test", name=dataset_name, r3_diffuser=r3_diffuser)
testloader = DataLoader(
    trainset, batch_size=64, shuffle=True, num_workers=0, collate_fn=collate_fn
)




'''
R3 SDE Inference
'''
def inference(model, trans_t, t, dt, noise_scale=1.0):
    with torch.no_grad():
        t_index = t <= r3_conf.min_t
        t[t_index] = r3_conf.min_t
        input = torch.cat([trans_t, t[:,None]],dim=-1)
        trans_score = model(input)

        trans_t = trans_t.cpu().numpy()
        trans_score = trans_score.cpu().numpy()

        trans_t_1 = r3_diffuser.reverse(
            x_t= trans_t,
            score_t= trans_score,
            t=t[0].item(),
            dt=dt.item(),
            center=True,
        )
    return torch.tensor(trans_t_1).float().to(device)


# Main Loop
def main_loop(model, optimizer, run_idx=0, num_epochs=150, display=True):
    losses = []
    w1ds = []

    final_traj = None
    for epoch in tqdm(range(num_epochs)):
        if (epoch % 10) == 0:
            n_test = testset.data.shape[0]
            traj = torch.tensor(r3_diffuser.sample_ref(n_test)).float().to(device)
            for t in torch.linspace(1, 0, 200):
                t = torch.tensor([t]).to(device).repeat(n_test).requires_grad_(True)
                dt = torch.tensor([1 / 200]).to(device)
                traj = inference(model, traj, t, dt)
            final_traj = traj.squeeze().cpu().numpy()
            test_data = testset.data.squeeze()
            w_d1 = wasserstein_distance_nd(final_traj, test_data)
            w1ds.append(w_d1)

        if display and (epoch % 100)==0:
            plot_r3(final_traj, title='$R^3$ Score Matching')
            plt.savefig(os.path.join(savedir, f"dataset_{dataset_name.split('.')[0]}_run{run_idx}_epoch{epoch}.jpg"))
            plt.show()
            print('wassterstein-1 distance:', w_d1)

        # Train Model
        for _, batch in enumerate(trainloader):
            for key in batch.keys():
                batch[key] = torch.tensor(batch[key]).to(device)
            trans_t = batch['trans_t'].float()
            input = torch.cat([trans_t, batch['t']],dim=-1).float()
            pred_trans_score = model(input)


            trans_mse = (batch['trans_score'].float()- pred_trans_score) ** 2
            trans_score_scaling = batch['trans_score_scaling']
            trans_loss = torch.sum(
                trans_mse / trans_score_scaling[:, None, None] ** 2)
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
num_runs = 1
for i in range(num_runs):
    print('doing run ', i)
    dim = 3  # network ouput is 3 dimensional (trans_vec matrix)
    model = MLP(dim=dim, time_varying=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    model, losses, w1ds = main_loop(model, optimizer, run_idx=i, num_epochs=1000, display=True)

    w1ds_runs.append(w1ds)
    losses_runs.append(losses)

losses_runs = np.array(losses_runs)
w1ds_runs = np.array(w1ds_runs)

np.save(os.path.join(savedir, f"{dataset_name.split('.')[0]}_losses.npy"), losses_runs)
np.save(os.path.join(savedir, f"{dataset_name.split('.')[0]}_w1ds.npy",), w1ds_runs)
