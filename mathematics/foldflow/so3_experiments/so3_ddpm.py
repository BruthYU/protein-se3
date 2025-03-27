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

os.environ['GEOMSTATS_BACKEND'] = 'pytorch'
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
SO3 DDPM Scheduler
'''
so3_scheduler = SO3_DDPM_Scheduler()

'''
Load Dataset
'''
dataset_name = "bunny_group.npy"


trainset = DDPM_Dataset(split="train", name=dataset_name)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=0)


testset = DDPM_Dataset(split="test", name=dataset_name)
testloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=0)


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
            traj = torch.tensor(so3_diffuser.sample_ref(n_test)).to(device)

            for t in torch.linspace(1, 0, 500):
                t = torch.tensor([t]).to(device).repeat(n_test).requires_grad_(True)
                dt = torch.tensor([1 / 500]).to(device)
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
            for key in batch.keys():
                batch[key] = torch.tensor(batch[key]).to(device)
            rot_t = batch['rot_t']
            input = torch.cat([rot_t, batch['t'][:,None]],dim=-1)
            pred_rot_score = model(input)


            rot_mse = (batch['rot_score']- pred_rot_score) ** 2
            rot_score_scaling = batch['rot_score_scaling']
            rot_loss = torch.sum(
                rot_mse / rot_score_scaling[:, None, None] ** 2)
            rot_loss.backward()

            optimizer.step()
            losses.append(rot_loss.detach().cpu().numpy())
            optimizer.zero_grad()


    return model, np.array(losses), np.array(w1ds), np.array(w2ds)



dim = 3  # network ouput is 3 dimensional (rot_vec matrix)
model = MLP(dim=dim, diffuser=so3_diffuser, time_varying=True).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
model, losses, w1ds, w2ds = main_loop(model, optimizer, num_epochs=1000, display=True)


