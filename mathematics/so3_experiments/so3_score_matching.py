import os
import numpy as np
import torch



os.environ['GEOMSTATS_BACKEND'] = 'pytorch'
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
from mathematics.so3_experiments.models.models import MLP
from torch.utils.data import DataLoader
from data.datasets import SpecialOrthogonalGroup, SDE_Dataset, concat_np_features

from geomstats._backend import _backend_config as _config
from lightning.data.framediff.so3_diffuser import SO3Diffuser
from lightning.data.framediff.so3_utils import Log, Exp

_config.DEFAULT_DTYPE = torch.cuda.FloatTensor
savedir = "results/score_matching"
os.makedirs(savedir, exist_ok=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)



'''
Load toy dataset
'''
dataset_name = "bunny_group.npy"
data = np.load(f'data/{dataset_name}',allow_pickle=True)
print('size of toy dataset: ', len(data))
fig = plot_so3(data, adjust=True)
plt.savefig(f"{savedir}/{dataset_name.split('.')[0]}.png", dpi=300)
plt.show()

'''
SO3 Diffuser
'''
from omegaconf import OmegaConf
so3_conf = OmegaConf.load('./config/score_matching.yaml')
so3_diffuser = SO3Diffuser(so3_conf.so3)

'''
Load Dataset
'''

collate_fn = lambda x: concat_np_features(x, add_batch_dim=True)

trainset = SDE_Dataset(split="train", name=dataset_name, data_conf=so3_conf.so3, so3_diffuser=so3_diffuser)
trainloader = DataLoader(
    trainset, batch_size=64, shuffle=True, num_workers=0, collate_fn=collate_fn
)


testset = SDE_Dataset(split="test", name=dataset_name, data_conf=so3_conf.so3, so3_diffuser=so3_diffuser)
testloader = DataLoader(
    trainset, batch_size=64, shuffle=True, num_workers=0, collate_fn=collate_fn
)


'''
SDE Inference
'''
def inference(model, rot_t, t, dt, noise_scale=1.0):
    # rot_t rotation vector
    with torch.no_grad():
        t_index = t <= so3_conf.so3.min_t
        t[t_index] = so3_conf.so3.min_t
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
def main_loop(model, optimizer, run_idx=0, num_epochs=150, display=True):
    losses = []
    w1ds = []
    w2ds = []


    final_traj = None
    for epoch in tqdm(range(num_epochs)):
        if (epoch % 10) == 0:
            n_test = testset.data.shape[0]
            traj = torch.tensor(so3_diffuser.sample_ref(n_test)).to(device)

            for t in torch.linspace(1, 0, 200):
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


        if display and (epoch % 100) == 0:
            plot_so3(final_traj)
            plt.savefig(os.path.join(savedir,  f"dataset_{dataset_name.split('.')[0]}_run{run_idx}_epoch{epoch}.jpg"))
            plt.show()
            print('wassterstein-1 distance:', w_d1)
            print('wassterstein-2 distance:', w_d2)


        for _, batch in enumerate(trainloader):
            for key in batch.keys():
                batch[key] = torch.tensor(batch[key]).to(device)
            rot_t = batch['rot_t']
            input = torch.cat([rot_t, batch['t']],dim=-1)
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



# dim = 3  # network ouput is 3 dimensional (rot_vec matrix)
# model = MLP(dim=dim, time_varying=True).to(device)
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
    model = MLP(dim=dim, time_varying=True).double().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    model, losses, w1ds, w2ds = main_loop(model, optimizer, run_idx=i, num_epochs=1000, display=True)

    w1ds_runs.append(w1ds)
    w2ds_runs.append(w2ds)
    losses_runs.append(losses)

losses_runs = np.array(losses_runs)
w1ds_runs = np.array(w1ds_runs)
w2ds_runs = np.array(w2ds_runs)

np.save(os.path.join(savedir, f"{dataset_name.split('.')[0]}_losses.npy", losses))
np.save(os.path.join(savedir, f"{dataset_name.split('.')[0]}_w1ds.npy", w1ds))
np.save(os.path.join(savedir, f"{dataset_name.split('.')[0]}_w2ds.npy", w2ds))

