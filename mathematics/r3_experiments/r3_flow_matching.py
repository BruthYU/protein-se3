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
from utils.plotting import plot_r3
from scipy.stats import wasserstein_distance_nd
from lightning.data.foldflow.so3_helpers import norm_SO3, expmap, rotmat_to_rotvec
from lightning.data.foldflow.condflowmatcher import ConditionalFlowMatcher
from mathematics.r3_experiments.models.models import MLP
from torch.utils.data import DataLoader
from data.datasets import R3Dataset
from geomstats._backend import _backend_config as _config

_config.DEFAULT_DTYPE = torch.cuda.FloatTensor
savedir = "results/flow_matching"
os.makedirs(savedir, exist_ok=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# Load toy dataset
dataset_name = "lorenz.npy"
data = np.load(f'data/{dataset_name}',allow_pickle=True)
print('size of toy dataset: ', len(data))
fig = plot_r3(data, title="Target R3 Distribution A")
plt.savefig(f"{savedir}/{dataset_name.split('.')[0]}.png", dpi=300)
plt.show()

# Load Dataset
trainset = R3Dataset(split="train", name=dataset_name)
trainloader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=0)
testset = R3Dataset(split="test", name=dataset_name)
testloader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=0)



'''
CFM
'''
R3FM = ConditionalFlowMatcher()


'''
loss and ODE inference
'''


def sample_ref(n_samples: float = 1):
    return np.random.normal(size=(n_samples, 3))

def loss_fn(v, u, x):
    # res = v - u
    # mse = torch.nn.MSELoss(reduction='mean')
    # loss = mse(x, res)
    mse = torch.nn.MSELoss(reduction='sum')
    loss = mse(v, u)
    return loss

def inference(model, xt, t, dt, center=False):
    with torch.no_grad():
        vt = model(torch.cat([xt, t[:, None]], dim=-1))
        perturb = vt * dt
        x_t_1 = xt + perturb
    return x_t_1





'''
Training Loop
'''
def main_loop(model, optimizer, run_idx=0, num_epochs=150, display=True):
    losses = []
    w1ds = []

    global_step = 0

    for epoch in tqdm(range(num_epochs)):

        final_traj = None
        if epoch % 10 == 0:
            n_test = testset.data.shape[0]
            traj = torch.randn(size=(n_test, 3)).to(device)
            for t in torch.linspace(0, 1, 200):
                t = torch.tensor([t]).to(device).repeat(n_test).requires_grad_(True)
                dt = torch.tensor([1 / 200]).to(device)
                traj = inference(model, traj, t, dt)
            final_traj = traj.squeeze().cpu().numpy()
            test_data = testset.data.squeeze()
            w_d1 = wasserstein_distance_nd(final_traj, test_data)
            w1ds.append(w_d1)
            # P = torch.tensor(testset.data).to(device).double().detach()
            # P = rotmat_to_rotvec(P)
            # Q = rotmat_to_rotvec(final_traj.detach())
            # kl_div = torch.nn.functional.kl_div(Q.softmax(-1).log(), P.softmax(-1), reduction='sum')



        if display and (epoch % 100)==0:
            plot_r3(final_traj, title='R3 Flow Matching')
            plt.savefig(os.path.join(savedir, f"dataset_{dataset_name.split('.')[0]}_run{run_idx}_epoch{epoch}.jpg"))
            plt.show()
            print('wassterstein-1 distance:', w_d1)

        for _, data in enumerate(trainloader):
            optimizer.zero_grad()
            x1 = data.squeeze()
            x0 = torch.randn(size=(len(x1), 3))

            t, xt, ut = R3FM.sample_location_and_conditional_flow_simple(x0.double(), x1.double())
            t = t.to(device)
            xt = xt.squeeze().to(device)
            ut = ut.squeeze().to(device)

            vt = model(torch.cat([xt, t[:, None]], dim=-1))

            loss = loss_fn(vt, ut, xt)
            losses.append(loss.detach().cpu().numpy())

            loss.backward()
            optimizer.step()



    return model, np.array(losses), np.array(w1ds)




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
    model = MLP(dim=dim, time_varying=True).double().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    model, losses, w1ds = main_loop(model, optimizer, run_idx=i, num_epochs=1000, display=True)

    w1ds_runs.append(w1ds)
    losses_runs.append(losses)

losses_runs = np.array(losses_runs)
w1ds_runs = np.array(w1ds_runs)


np.save(os.path.join(savedir, f"{dataset_name.split('.')[0]}_losses.npy"), losses_runs)
np.save(os.path.join(savedir, f"{dataset_name.split('.')[0]}_w1ds.npy",), w1ds_runs)
