import sys
sys.path.append("..")
import os
os.environ['GEOMSTATS_BACKEND'] = 'pytorch'
import numpy as np
import torch
from einops import rearrange
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')
from scipy.spatial.transform import Rotation
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from utils.plotting import plot_so3
from utils.optimal_transport import so3_wasserstein as wasserstein
from lightning.data.foldflow.so3_helpers import norm_SO3, expmap, rotmat_to_rotvec
from lightning.data.foldflow.so3_condflowmatcher import SO3ConditionalFlowMatcher
from mathematics.so3_experiments.models.models import PMLP
from torch.utils.data import DataLoader
from data.datasets import SpecialOrthogonalGroup
from geomstats._backend import _backend_config as _config

_config.DEFAULT_DTYPE = torch.cuda.FloatTensor
savedir = "models/so3_synthetic"
os.makedirs(savedir, exist_ok=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

root = "data"
dataset_name = "bunny_group.npy"

'''
Load Dataset
'''
data = np.load(f'{root}/{dataset_name}')
print('size of toy dataset: ', len(data))
fig = plot_so3(data, adjust=True)
plt.savefig('figs/so3_synthetic_data.png', dpi=300)
plt.show()

trainset = SpecialOrthogonalGroup(name=dataset_name, split="train")
trainloader = DataLoader(
    trainset, batch_size=64, shuffle=True, num_workers=0
)

testset = SpecialOrthogonalGroup(name=dataset_name, split="test")
testloader = DataLoader(
    testset, batch_size=64, shuffle=False, num_workers=0
)


'''
CFM
'''
so3_group = SpecialOrthogonal(n=3, point_type="matrix")
FM = SO3ConditionalFlowMatcher(manifold=so3_group)


'''
loss and ODE inference
'''
def loss_fn(v, u, x):
    res = v - u
    norm = norm_SO3(x, res) # norm-squared on SO(3)
    loss = torch.mean(norm, dim=-1)
    return loss

def inference(model, xt, t, dt):
    with torch.no_grad():
        vt = model(torch.cat([xt, t[:, None]], dim=-1)) # vt on the tanget of xt
        vt = rearrange(vt, 'b (c d) -> b c d', c=3, d=3)
        xt = rearrange(xt, 'b (c d) -> b c d', c=3, d=3)
        xt_new = expmap(xt, vt * dt)                   # expmap to get the next point
    return rearrange(xt_new, 'b c d -> b (c d)', c=3, d=3)

dim = 9 # network ouput is 9 dimensional (3x3 matrix)
# MLP with a projection at the end, projection on to the tanget space of the manifold
model = PMLP(dim=dim, time_varying=True).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)



'''
Training Loop
'''
def main_loop(model, optimizer, num_epochs=150, display=True):
    losses = []
    w1ds = []
    w2ds = []
    global_step = 0

    for epoch in range(num_epochs):

        if display:
            progress_bar = tqdm(total=len(trainloader))
            progress_bar.set_description(f"Epoch {epoch}")

        if epoch % 100 == 0:
            n_test = testset.data.shape[0]
            traj = torch.tensor(Rotation.random(n_test).as_matrix()).to(device).reshape(-1, 9)
            for t in torch.linspace(0, 1, 200):
                t = torch.tensor([t]).to(device).repeat(n_test).requires_grad_(True)
                dt = torch.tensor([1 / 200]).to(device)
                traj = inference(model, traj, t, dt)
            final_traj = rearrange(traj, 'b (c d) -> b c d', c=3, d=3)

            # P = torch.tensor(testset.data).to(device).double().detach()
            # P = rotmat_to_rotvec(P)
            # Q = rotmat_to_rotvec(final_traj.detach())
            # kl_div = torch.nn.functional.kl_div(Q.softmax(-1).log(), P.softmax(-1), reduction='sum')

            w_d1 = wasserstein(torch.tensor(testset.data).to(device).double().detach(), final_traj.detach(), power=1)
            w_d2 = wasserstein(torch.tensor(testset.data).to(device).double().detach(), final_traj.detach(), power=2)
            w1ds.append(w_d1)
            w2ds.append(w_d2)

            if display:
                plot_so3(final_traj, adjust=True)
                plt.show()
                print('wassterstein-1 distance:', w_d1)
                print('wassterstein-2 distance:', w_d2)

        for _, data in enumerate(trainloader):
            optimizer.zero_grad()
            x1 = data
            x0 = torch.tensor(Rotation.random(x1.size(0)).as_matrix())

            t, xt, ut = FM.sample_location_and_conditional_flow_simple(x0.double(), x1.double())
            t = t.to(device)
            xt = xt.to(device)
            ut = ut.to(device)

            vt = model(torch.cat([rearrange(xt, 'b c d -> b (c d)', c=3, d=3), t[:, None]], dim=-1))
            vt = rearrange(vt, 'b (c d) -> b c d', c=3, d=3)

            loss = loss_fn(vt, ut, xt)
            losses.append(loss.detach().cpu().numpy())

            loss.backward()
            optimizer.step()

            if display:
                progress_bar.update(1)
                logs = {"loss": loss.detach().item(), "step": global_step}
                progress_bar.set_postfix(**logs)
                global_step += 1

    return model, np.array(losses), np.array(w1ds), np.array(w2ds)


model, losses, w1s, w2s = main_loop(model, optimizer, num_epochs=1000, display=True)

