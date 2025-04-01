import os

import matplotlib.pyplot as plt
from geomstats._backend import _backend_config as _config
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.datasets import DDPM_Dataset
from mathematics.so3_experiments.models.models import UMLP
from utils.ddpm_utils import *
from utils.optimal_transport import so3_wasserstein as wasserstein
from utils.plotting import plot_so3

torch.manual_seed(0)

os.environ['GEOMSTATS_BACKEND'] = 'pytorch'
_config.DEFAULT_DTYPE = torch.cuda.FloatTensor
savedir = "results/ddpm"
os.makedirs(savedir, exist_ok=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
so3_conf = OmegaConf.load('./config/ddpm.yaml')
n_timestep = so3_conf.diffusion.n_timestep

# Load toy dataset
dataset_name = "bunny_group.npy"
data = np.load(f'data/{dataset_name}',allow_pickle=True)
print('size of toy dataset: ', len(data))
fig = plot_so3(data, adjust=True,title="Target Distribution")
plt.savefig(f"{savedir}/{dataset_name.split('.')[0]}.png", dpi=300)
plt.show()


# SO3 DDPM Schedulera
Rotation_DDPM = RotationTransition(num_steps=n_timestep)


# Load Dataset

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
    rot_t, _ = Rotation_DDPM.add_noise(rot_0, t) # []
    return rot_t


# DDPM Inference
def inference(model, rot_t, t):
    # rot_t rotation vector
    with torch.no_grad():
        t_tensor = t / n_timestep
        input = torch.cat([rot_t, t_tensor[:, None, None]], dim=-1)
        # rotvec_next, rotmatrix_next = model(input)
        rotvec_next = model(input)


        # Compute posterior
        rot_next = Rotation_DDPM.denoise(rotvec_next, t.cpu())
    return rot_next.to(device)


# Main Loop
def main_loop(model, optimizer, run_idx=0, num_epochs=150, display=True):
    losses = []
    w1ds = []
    w2ds = []

    final_traj = None
    for epoch in tqdm(range(num_epochs)):
        if (epoch % 10) == 0:
            n_test = testset.data.shape[0]
            # traj = torch.tensor(Rotation.random(n_test).as_rotvec()).to(device)

            traj = random_normal_so3(torch.tensor(n_timestep)[None].expand(n_test,1),
                                     Rotation_DDPM.angular_distrib_fwd)
            # traj = random_uniform_so3([n_test, 1])
            traj = traj.double().to(device)
            steps = range(n_timestep, 0, -1)
            for t in steps:
                t = torch.tensor([t]).to(device).repeat(n_test)
                traj = inference(model, traj, t)
            final_traj = so3vec_to_rotation(traj).squeeze()
            final_traj = final_traj.cpu().numpy()
            w_d1 = wasserstein(torch.tensor(testset.data).to(device).double().detach(),
                               torch.tensor(final_traj).to(device).double().detach(), power=1)
            w_d2 = wasserstein(torch.tensor(testset.data).to(device).double().detach(),
                               torch.tensor(final_traj).to(device).double().detach(), power=2)
            w1ds.append(w_d1)
            w2ds.append(w_d2)
        if display and (epoch % 100)==0:
            plot_so3(final_traj, adjust=True, title='SO(3) DDPM')
            plt.savefig(os.path.join(savedir, f"dataset_{dataset_name.split('.')[0]}_run{run_idx}_epoch{epoch}.jpg"))
            # plt.show()
            print('wassterstein-1 distance:', w_d1)
            print('wassterstein-2 distance:', w_d2)

        # Train Model
        for _, data in enumerate(trainloader):

            rotmatrix_0 = torch.tensor(data).double().to(device)
            rotvec_0 = rotation_to_so3vec(rotmatrix_0)
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