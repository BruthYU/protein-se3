import inspect
import random
import torch
import importlib
import torch.optim.lr_scheduler as lrs
import pytorch_lightning as pl
import os
import torch.nn as nn
import copy
from lightning.data.foldflow.so3_helpers import hat_inv, pt_to_identity
from lightning.data.foldflow import dataloader as du
from preprocess.tools import all_atom
from einops import rearrange
from lightning.data.foldflow import se3_fm
from lightning.model.foldflow import network
from evaluate.openfold.utils import rigid_utils as ru
from .analysis import metrics
from .analysis import utils as au
import numpy as np
import pandas as pd
import tree
from lightning.model.foldflow import network
from lightning.data.foldflow import se3_fm

import copy
import logging
import time
LOG = logging.getLogger(__name__)

# TODO foldflow-model
# TODO eval_fn
class foldflow_Lightning_Model(pl.LightningModule):
    def __init__(self, conf):
        super().__init__()
        self.save_hyperparameters()
        self.model_conf = conf.model
        self.fm_conf = conf.flow_matcher
        self.exp_conf = conf.experiment
        self.data_conf = conf.dataset

        self.flow_matcher = se3_fm.SE3FlowMatcher(self.fm_conf)
        self.model = network.VectorFieldNetwork(self.model_conf, self.flow_matcher)
        self.validation_step_outputs = []


    def forward(self, batch, cond):
        model_out = self.model(batch)
        return model_out

    def training_step(self, batch, batch_idx, **kwargs):
        loss, aux_data = self.loss_fn(batch)
        # self.log("global_step", self.global_step, on_step=True, on_epoch=True, prog_bar=True)
        # self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        log_info = {
            "train_loss": loss,
            "rot_loss": aux_data["rot_loss"],
            "trans_loss": aux_data["trans_loss"],
            "bb_atom_loss": aux_data["bb_atom_loss"],
            "dist_mat_loss": aux_data["dist_mat_loss"],
        }
        self.log_dict(log_info, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        eval_fn_output = self.eval_fn(batch, batch_idx, noise_scale=self.exp_conf.noise_scale)
        self.log("tm_score", np.mean([x['tm_score'] for x in eval_fn_output]), on_epoch=True, prog_bar=True)
        self.validation_step_outputs.append(eval_fn_output)

    def on_validation_epoch_end(self) -> None:
        ckpt_eval_metrics = []
        for batch_eval_metrics in self.validation_step_outputs:
            ckpt_eval_metrics.extend(batch_eval_metrics)
        eval_metrics_csv_path = os.path.join(self.exp_conf.eval_dir, f'epoch_{self.current_epoch}', f"validation_rank_{self.local_rank}_metrics.csv")
        ckpt_eval_metrics = pd.DataFrame(ckpt_eval_metrics)
        ckpt_eval_metrics.to_csv(eval_metrics_csv_path, index=False)
        self.validation_step_outputs.clear()



    def get_schedular(self, optimizer, lr_scheduler='onecycle'):
        if lr_scheduler == 'step':
            scheduler = lrs.StepLR(optimizer,
                                   step_size=self.exp_conf.lr_decay_steps,
                                   gamma=self.exp_conf.lr_decay_rate)
        elif lr_scheduler == 'cosine':
            scheduler = lrs.CosineAnnealingLR(optimizer,
                                              T_max=self.exp_conf.lr_decay_steps,
                                              eta_min=self.exp_conf.lr_decay_min_lr)
        elif lr_scheduler == 'onecycle':
            scheduler = lrs.OneCycleLR(optimizer, max_lr=self.exp_conf.learning_rate, steps_per_epoch=self.exp_conf.steps_per_epoch,
                                       epochs=self.exp_conf.num_epoch, three_phase=False)

        elif lr_scheduler == 'LambdaLR':
            def lr_lambda(current_step):
                warmup_steps = self.exp_conf.warmup_steps
                if current_step < self.exp_conf.warmup_steps:
                    # Linearly increase learning rate
                    return float(current_step) / float(max(1, warmup_steps))
                else:
                    # After warmup, apply other schedule, e.g., constant
                    return 1.0

            scheduler = lrs.LambdaLR(optimizer, lr_lambda=lr_lambda)
            # scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=10, max_epochs=40)

        else:
            raise ValueError('Invalid lr_scheduler type!')

        return scheduler

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.exp_conf.learning_rate)
        schedular = self.get_schedular(optimizer, self.exp_conf.lr_scheduler)
        return [optimizer], [{"scheduler": schedular, "interval": "step"}]

    def lr_scheduler_step(self, *args, **kwargs):
        scheduler = self.lr_schedulers()
        scheduler.step()

    def self_conditioning(self, batch):
        model_sc = self.model(batch)
        batch["sc_ca_t"] = model_sc["rigids"][..., 4:]
        return batch

    def loss_fn(self, batch):
        """Computes loss and auxiliary data.

        Args:
            batch: Batched data.
            model_out: Output of model ran on batch.

        Returns:
            loss: Final training loss scalar.
            aux_data: Additional logging data.
        """
        if (
            self.model_conf.embed.embed_self_conditioning
            and self.global_step % 2 == 1
        ):
            # if self.model_conf.embed.embed_self_conditioning and random.random() > 0.5:
            with torch.no_grad():
                batch = self.self_conditioning(batch)

        _, gt_rot_u_t = self.flow_matcher._so3_fm.vectorfield(
            batch["rot_vectorfield"], batch["rot_t"], batch["t"]
        )


        model_out = self.model(batch)
        bb_mask = batch["res_mask"]
        flow_mask = 1 - batch["fixed_mask"]
        loss_mask = bb_mask * flow_mask
        batch_size, num_res = bb_mask.shape

        gt_trans_u_t = batch["trans_vectorfield"]
        rot_vectorfield_scaling = batch["rot_vectorfield_scaling"]
        trans_vectorfield_scaling = batch["trans_vectorfield_scaling"]
        batch_loss_mask = torch.any(bb_mask, dim=-1)

        pred_rot_v_t = model_out["rot_vectorfield"] * flow_mask[..., None, None]
        pred_trans_v_t = model_out["trans_vectorfield"] * flow_mask[..., None]

        # Translation vectorfield loss
        trans_vectorfield_mse = (gt_trans_u_t - pred_trans_v_t) ** 2 * loss_mask[
            ..., None
        ]
        trans_vectorfield_loss = torch.sum(
            trans_vectorfield_mse / trans_vectorfield_scaling[:, None, None] ** 2,
            dim=(-1, -2),
        ) / (loss_mask.sum(dim=-1) + 1e-10)

        # Translation x0 loss
        gt_trans_x0 = batch["rigids_0"][..., 4:] * self.exp_conf.coordinate_scaling
        pred_trans_x0 = model_out["rigids"][..., 4:] * self.exp_conf.coordinate_scaling
        trans_x0_loss = torch.sum(
            (gt_trans_x0 - pred_trans_x0) ** 2 * loss_mask[..., None], dim=(-1, -2)
        ) / (loss_mask.sum(dim=-1) + 1e-10)

        trans_loss = trans_vectorfield_loss * (
            batch["t"] > self.exp_conf.trans_x0_threshold
        ) + trans_x0_loss * (batch["t"] <= self.exp_conf.trans_x0_threshold)
        trans_loss *= self.exp_conf.trans_loss_weight
        trans_loss *= int(self.fm_conf.flow_trans)

        # Rotation loss
        # gt_rot_u_t and pred_rot_v_t are matrices convert
        t_shape = batch["rot_t"].shape[0]
        rot_t = rearrange(batch["rot_t"], "t n c d -> (t n) c d", c=3, d=3).double()
        gt_rot_u_t = rearrange(gt_rot_u_t, "t n c d -> (t n) c d", c=3, d=3)
        pred_rot_v_t = rearrange(pred_rot_v_t, "t n c d -> (t n) c d", c=3, d=3)
        try:
            rot_t = rot_t.double()
            gt_at_id = pt_to_identity(rot_t, gt_rot_u_t)
            gt_rot_u_t = hat_inv(gt_at_id)
            pred_at_id = pt_to_identity(rot_t, pred_rot_v_t)
            pred_rot_v_t = hat_inv(pred_at_id)
        except ValueError as e:
            LOG.info(
                f"Skew symmetric error gt {((gt_at_id + gt_at_id.transpose(-1, -2))**2).mean()} "
                f"pred {((pred_at_id + pred_at_id.transpose(-1, -2))**2).mean()} Skipping rot loss"
            )
            gt_rot_u_t = torch.zeros_like(rot_t[..., 0])
            pred_rot_v_t = torch.zeros_like(rot_t[..., 0])

        gt_rot_u_t = rearrange(gt_rot_u_t, "(t n) c -> t n c", t=t_shape, c=3)
        pred_rot_v_t = rearrange(pred_rot_v_t, "(t n) c -> t n c", t=t_shape, c=3)

        if self.exp_conf.separate_rot_loss:
            gt_rot_angle = torch.norm(gt_rot_u_t, dim=-1, keepdim=True)
            gt_rot_axis = gt_rot_u_t / (gt_rot_angle + 1e-6)

            pred_rot_angle = torch.norm(pred_rot_v_t, dim=-1, keepdim=True)
            pred_rot_axis = pred_rot_v_t / (pred_rot_angle + 1e-6)

            # Separate loss on the axis
            axis_loss = (gt_rot_axis - pred_rot_axis) ** 2 * loss_mask[..., None]
            axis_loss = torch.sum(axis_loss, dim=(-1, -2)) / (
                loss_mask.sum(dim=-1) + 1e-10
            )

            # Separate loss on the angle
            angle_loss = (gt_rot_angle - pred_rot_angle) ** 2 * loss_mask[..., None]
            angle_loss = torch.sum(
                angle_loss / rot_vectorfield_scaling[:, None, None] ** 2, dim=(-1, -2)
            ) / (loss_mask.sum(dim=-1) + 1e-10)
            angle_loss *= self.exp_conf.rot_loss_weight
            angle_loss *= batch["t"] > self.exp_conf.rot_loss_t_threshold
            rot_loss = angle_loss + axis_loss
        else:
            rot_mse = (gt_rot_u_t - pred_rot_v_t) ** 2 * loss_mask[..., None]
            rot_loss = torch.sum(
                rot_mse / rot_vectorfield_scaling[:, None, None] ** 2,
                dim=(-1, -2),
            ) / (loss_mask.sum(dim=-1) + 1e-10)
            rot_loss *= self.exp_conf.rot_loss_weight
            rot_loss *= batch["t"] > self.exp_conf.rot_loss_t_threshold
        rot_loss *= int(self.fm_conf.flow_rot)

        # Backbone atom loss
        pred_atom37 = model_out["atom37"][:, :, :5]
        gt_rigids = ru.Rigid.from_tensor_7(batch["rigids_0"].type(torch.float32))
        gt_psi = batch["torsion_angles_sin_cos"][..., 2, :]
        gt_atom37, atom37_mask, _, _ = all_atom.compute_backbone(gt_rigids, gt_psi)
        gt_atom37 = gt_atom37[:, :, :5]
        atom37_mask = atom37_mask[:, :, :5]

        gt_atom37 = gt_atom37.to(pred_atom37.device)
        atom37_mask = atom37_mask.to(pred_atom37.device)
        bb_atom_loss_mask = atom37_mask * loss_mask[..., None]
        bb_atom_loss = torch.sum(
            (pred_atom37 - gt_atom37) ** 2 * bb_atom_loss_mask[..., None],
            dim=(-1, -2, -3),
        ) / (bb_atom_loss_mask.sum(dim=(-1, -2)) + 1e-10)
        bb_atom_loss *= self.exp_conf.bb_atom_loss_weight
        bb_atom_loss *= batch["t"] < self.exp_conf.bb_atom_loss_t_filter
        bb_atom_loss *= self.exp_conf.aux_loss_weight

        # Pairwise distance loss
        gt_flat_atoms = gt_atom37.reshape([batch_size, num_res * 5, 3])
        gt_pair_dists = torch.linalg.norm(
            gt_flat_atoms[:, :, None, :] - gt_flat_atoms[:, None, :, :], dim=-1
        )
        pred_flat_atoms = pred_atom37.reshape([batch_size, num_res * 5, 3])
        pred_pair_dists = torch.linalg.norm(
            pred_flat_atoms[:, :, None, :] - pred_flat_atoms[:, None, :, :], dim=-1
        )

        flat_loss_mask = torch.tile(loss_mask[:, :, None], (1, 1, 5))
        flat_loss_mask = flat_loss_mask.reshape([batch_size, num_res * 5])
        flat_res_mask = torch.tile(bb_mask[:, :, None], (1, 1, 5))
        flat_res_mask = flat_res_mask.reshape([batch_size, num_res * 5])

        gt_pair_dists = gt_pair_dists * flat_loss_mask[..., None]
        pred_pair_dists = pred_pair_dists * flat_loss_mask[..., None]
        pair_dist_mask = flat_loss_mask[..., None] * flat_res_mask[:, None, :]

        # No loss on anything >6A
        proximity_mask = gt_pair_dists < 6
        pair_dist_mask = pair_dist_mask * proximity_mask

        dist_mat_loss = torch.sum(
            (gt_pair_dists - pred_pair_dists) ** 2 * pair_dist_mask, dim=(1, 2)
        )
        dist_mat_loss /= torch.sum(pair_dist_mask, dim=(1, 2)) - num_res
        dist_mat_loss *= self.exp_conf.dist_mat_loss_weight
        dist_mat_loss *= batch["t"] < self.exp_conf.dist_mat_loss_t_filter
        dist_mat_loss *= self.exp_conf.aux_loss_weight

        final_loss = rot_loss + trans_loss + bb_atom_loss + dist_mat_loss

        def normalize_loss(x):
            return x.sum() / (batch_loss_mask.sum() + 1e-10)

        aux_data = {
            "batch_train_loss": final_loss,
            "batch_rot_loss": rot_loss,
            "batch_trans_loss": trans_loss,
            "batch_bb_atom_loss": bb_atom_loss,
            "batch_dist_mat_loss": dist_mat_loss,
            "total_loss": normalize_loss(final_loss),
            "rot_loss": normalize_loss(rot_loss),
            "trans_loss": normalize_loss(trans_loss),
            "bb_atom_loss": normalize_loss(bb_atom_loss),
            "dist_mat_loss": normalize_loss(dist_mat_loss),
            "examples_per_step": torch.tensor(batch_size),
            "res_length": torch.mean(torch.sum(bb_mask, dim=-1)),
        }



        assert final_loss.shape == (batch_size,)
        assert batch_loss_mask.shape == (batch_size,)
        return normalize_loss(final_loss), aux_data
    
    def eval_fn(
        self,
        batch,
        batch_idx,
        min_t=None,
        num_t=None,
        noise_scale=1.0,
        context=None,
    ):
        valid_feats = batch
        batch_eval_metrics = []

        res_mask = du.move_to_np(valid_feats["res_mask"].bool())
        fixed_mask = du.move_to_np(valid_feats["fixed_mask"].bool())
        aatype = du.move_to_np(valid_feats["aatype"])
        gt_prot = du.move_to_np(valid_feats["atom37_pos"])
        lmdbIndex = du.move_to_np(valid_feats['lmdbIndex'])
        batch_size = res_mask.shape[0]

        # Run inference
        infer_out = self.inference_fn(
            valid_feats,
            min_t=min_t,
            num_t=num_t,
            noise_scale=noise_scale,
            context=context,
        )
        final_prot = infer_out["prot_traj"][0]
        for i in range(batch_size):
            num_res = int(np.sum(res_mask[i]).item())
            unpad_fixed_mask = fixed_mask[i][res_mask[i]]
            unpad_flow_mask = 1 - unpad_fixed_mask
            unpad_prot = final_prot[i][res_mask[i]]
            unpad_gt_prot = gt_prot[i][res_mask[i]]
            unpad_gt_aatype = aatype[i][res_mask[i]]
            percent_flowed = np.sum(unpad_flow_mask) / num_res

            prot_path = os.path.join(
                    self.exp_conf.eval_dir,
                    f'epoch_{self.current_epoch}',
                    f'Rank{self.local_rank}_B{batch_idx}S{i}_lmdbIndex_{lmdbIndex[i]}_len_{num_res}.pdb')

            # Extract argmax predicted aatype
            saved_path = au.write_prot_to_pdb(
                unpad_prot,
                prot_path,
                b_factors=np.tile(1 - unpad_fixed_mask[..., None], 37) * 100,
            )


            try:
                # Including TM_Score
                sample_metrics = metrics.protein_metrics(
                    pdb_path=saved_path,
                    atom37_pos=unpad_prot,
                    gt_atom37_pos=unpad_gt_prot,
                    gt_aatype=unpad_gt_aatype,
                    flow_mask=unpad_flow_mask,
                )
            except ValueError as e:
                LOG.warning(
                    f"Failed evaluation of length {num_res} sample {i}: {e}"
                )
                continue

            sample_metrics['lmdb_csv_Index'] = lmdbIndex[i]
            sample_metrics["num_res"] = num_res
            sample_metrics["fixed_residues"] = np.sum(unpad_fixed_mask)
            sample_metrics["flowed_percentage"] = percent_flowed
            sample_metrics["sample_path"] = saved_path
            batch_eval_metrics.append(sample_metrics)

        return batch_eval_metrics

    def set_t_feats(self, feats, t, t_placeholder):
        feats["t"] = t * t_placeholder
        (
            rot_vectorfield_scaling,
            trans_vectorfield_scaling,
        ) = self.flow_matcher.vectorfield_scaling(t)
        feats["rot_vectorfield_scaling"] = rot_vectorfield_scaling * t_placeholder
        feats["trans_vectorfield_scaling"] = trans_vectorfield_scaling * t_placeholder
        return feats


    def inference_fn(
        self,
        data_init,
        num_t=None,
        min_t=None,
        center=True,
        aux_traj=False,
        self_condition=True,
        noise_scale=1.0,
        context=None,
    ):
        """Inference function.

        Args:
            data_init: Initial data values for sampling.
        """

        # Run reverse process.
        sample_feats = copy.deepcopy(data_init)
        device = sample_feats["rigids_t"].device
        if sample_feats["rigids_t"].ndim == 2:
            t_placeholder = torch.ones((1,)).to(device)
        else:
            t_placeholder = torch.ones((sample_feats["rigids_t"].shape[0],)).to(device)
        if num_t is None:
            num_t = self.data_conf.num_t
        if min_t is None:
            min_t = self.data_conf.min_t
        reverse_steps = np.linspace(min_t, 1.0, num_t)[::-1]
        dt = reverse_steps[0] - reverse_steps[1]
        # dt = 1/num_t
        all_rigids = [du.move_to_np(copy.deepcopy(sample_feats["rigids_t"]))]
        all_bb_prots = []
        all_trans_0_pred = []
        all_bb_0_pred = []
        with torch.no_grad():
            if self.model_conf.embed.embed_self_conditioning and self_condition:
                sample_feats = self.set_t_feats(
                    sample_feats, reverse_steps[0], t_placeholder
                )
                sample_feats = self.self_conditioning(sample_feats)
            for t in reverse_steps:

                sample_feats = self.set_t_feats(sample_feats, t, t_placeholder)
                model_out = self.model(sample_feats)
                rot_vectorfield = model_out["rot_vectorfield"]
                trans_vectorfield = model_out["trans_vectorfield"]
                rigid_pred = model_out["rigids"]
                if self.model_conf.embed.embed_self_conditioning:
                    sample_feats["sc_ca_t"] = rigid_pred[..., 4:]
                fixed_mask = sample_feats["fixed_mask"] * sample_feats["res_mask"]
                flow_mask = (1 - sample_feats["fixed_mask"]) * sample_feats["res_mask"]
                rots_t, trans_t, rigids_t = self.flow_matcher.reverse(
                    rigid_t=ru.Rigid.from_tensor_7(sample_feats["rigids_t"]),
                    rot_vectorfield=du.move_to_np(rot_vectorfield),
                    trans_vectorfield=du.move_to_np(trans_vectorfield),
                    flow_mask=du.move_to_np(flow_mask),
                    t=t,
                    dt=dt,
                    center=center,
                    noise_scale=noise_scale,
                )

                sample_feats["rigids_t"] = rigids_t.to_tensor_7().to(device)
                if aux_traj:
                    all_rigids.append(du.move_to_np(rigids_t.to_tensor_7()))

                # Calculate x0 prediction derived from vectorfield predictions.
                gt_trans_0 = sample_feats["rigids_t"][..., 4:]
                pred_trans_0 = rigid_pred[..., 4:]
                trans_pred_0 = (
                    flow_mask[..., None] * pred_trans_0
                    + fixed_mask[..., None] * gt_trans_0
                )
                psi_pred = model_out["psi"]
                if aux_traj:
                    atom37_0 = all_atom.compute_backbone(
                        ru.Rigid.from_tensor_7(rigid_pred), psi_pred
                    )[0]
                    all_bb_0_pred.append(du.move_to_np(atom37_0))
                    all_trans_0_pred.append(du.move_to_np(trans_pred_0))
                atom37_t = all_atom.compute_backbone(rigids_t, psi_pred)[0]
                all_bb_prots.append(du.move_to_np(atom37_t))

        # Flip trajectory so that it starts from t=0.
        # This helps visualization.
        flip = lambda x: np.flip(np.stack(x), (0,))
        all_bb_prots = flip(all_bb_prots)
        if aux_traj:
            all_rigids = flip(all_rigids)
            all_trans_0_pred = flip(all_trans_0_pred)
            all_bb_0_pred = flip(all_bb_0_pred)

        ret = {
            "prot_traj": all_bb_prots,
        }
        if aux_traj:
            ret["rigid_traj"] = all_rigids
            ret["trans_traj"] = all_trans_0_pred
            ret["psi_pred"] = psi_pred[None]
            ret["rigid_0_traj"] = all_bb_0_pred
        return ret


