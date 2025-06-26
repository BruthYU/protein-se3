from omegaconf import DictConfig, OmegaConf
import logging
import torch
import random
import numpy as np
import lightning_protein.data.rfdiffusion.denoiser as iu
from lightning_protein.model.rfdiffusion.util import writepdb_multi, writepdb
import os, time, pickle

def make_deterministic(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

class rfdiffusion_Sampler:
    def __init__(self, conf: DictConfig):

        self.log = logging.getLogger(__name__)

        if conf.inference.deterministic:
            make_deterministic()
        self.conf = conf

        self.selected_sampler = iu.sampler_selector(conf)

        # Check for available GPU and print result of check
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(torch.cuda.current_device())
            self.log.info(f"Found GPU with device_name {device_name}. Will run RFdiffusion on {device_name}")
        else:
            self.log.info("////////////////////////////////////////////////")
            self.log.info("///// NO GPU DETECTED! Falling back to CPU /////")
            self.log.info("////////////////////////////////////////////////")

        # Loop over number of designs to sample.
        design_startnum = self.selected_sampler.inf_conf.design_startnum




    def run_sampling(self):
        for i_des in range(self.selected_sampler.inf_conf.num_designs):
            if self.conf.inference.deterministic:
                make_deterministic(i_des)

            start_time = time.time()
            out_prefix = f"{self.selected_sampler.inf_conf.output_prefix}_{i_des}"
            mask_out_prefix = f"{self.selected_sampler.inf_conf.mask_output_prefix}_{i_des}"

            out_prefix = os.path.join(f"{self.selected_sampler.inf_conf.output_dir}", out_prefix)
            mask_out_prefix = os.path.join(f"{self.selected_sampler.inf_conf.output_dir}", mask_out_prefix)
            self.log.info(f"Making design {out_prefix}")
            self.log.info(f"Saving masks {mask_out_prefix}")

            # Duplicated PDB File
            if self.selected_sampler.inf_conf.cautious and os.path.exists(out_prefix + ".pdb"):
                self.log.info(
                    f"(cautious mode) Skipping this design because {out_prefix}.pdb already exists."
                )
                continue

            x_init, seq_init = self.selected_sampler.sample_init()
            denoised_xyz_stack = []
            px0_xyz_stack = []
            seq_stack = []
            plddt_stack = []

            x_t = torch.clone(x_init)
            seq_t = torch.clone(seq_init)
            # Loop over number of reverse diffusion time steps.
            for t in range(int(self.selected_sampler.t_step_input), self.selected_sampler.inf_conf.final_step - 1, -1):
                px0, x_t, seq_t, plddt = self.selected_sampler.sample_step(
                    t=t, x_t=x_t, seq_init=seq_t, final_step=self.selected_sampler.inf_conf.final_step
                )
                px0_xyz_stack.append(px0)
                denoised_xyz_stack.append(x_t)
                seq_stack.append(seq_t)
                plddt_stack.append(plddt[0])  # remove singleton leading dimension

            # Flip order for better visualization in pymol
            denoised_xyz_stack = torch.stack(denoised_xyz_stack)
            denoised_xyz_stack = torch.flip(
                denoised_xyz_stack,
                [
                    0,
                ],
            )
            px0_xyz_stack = torch.stack(px0_xyz_stack)
            px0_xyz_stack = torch.flip(
                px0_xyz_stack,
                [
                    0,
                ],
            )

            # For logging -- don't flip
            plddt_stack = torch.stack(plddt_stack)

            # Save outputs
            os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
            os.makedirs(os.path.dirname(mask_out_prefix), exist_ok=True)
            final_seq = seq_stack[-1]

            # Output glycines, except for motif region
            final_seq = torch.where(
                torch.argmax(seq_init, dim=-1) == 21, 7, torch.argmax(seq_init, dim=-1)
            )  # 7 is glycine

            bfacts = torch.ones_like(final_seq.squeeze())
            # make bfact=0 for diffused coordinates
            bfacts[torch.where(torch.argmax(seq_init, dim=-1) == 21, True, False)] = 0
            # pX0 last step
            out = f"{out_prefix}.pdb"
            mask_out = f"{mask_out_prefix}.npy"

            # save diffusion mask
            motif_mask = self.selected_sampler.diffusion_mask[0].numpy()
            np.save(mask_out, motif_mask)

            # Now don't output sidechains
            writepdb(
                out,
                denoised_xyz_stack[0, :, :4],
                final_seq,
                self.selected_sampler.binderlen,
                chain_idx=self.selected_sampler.chain_idx,
                bfacts=bfacts,
            )

            # # run metadata
            # trb = dict(
            #     config=OmegaConf.to_container(self.selected_sampler._conf, resolve=True),
            #     plddt=plddt_stack.cpu().numpy(),
            #     device=torch.cuda.get_device_name(torch.cuda.current_device())
            #     if torch.cuda.is_available()
            #     else "CPU",
            #     time=time.time() - start_time,
            # )
            # if hasattr(self.selected_sampler, "contig_map"):
            #     for key, value in self.selected_sampler.contig_map.get_mappings().items():
            #         trb[key] = value
            # with open(f"{out_prefix}.trb", "wb") as f_out:
            #     pickle.dump(trb, f_out)
            #
            # if self.selected_sampler.inf_conf.write_trajectory:
            #     # trajectory pdbs
            #     traj_prefix = (
            #             os.path.dirname(out_prefix) + "/traj/" + os.path.basename(out_prefix)
            #     )
            #     os.makedirs(os.path.dirname(traj_prefix), exist_ok=True)
            #
            #     out = f"{traj_prefix}_Xt-1_traj.pdb"
            #     writepdb_multi(
            #         out,
            #         denoised_xyz_stack,
            #         bfacts,
            #         final_seq.squeeze(),
            #         use_hydrogens=False,
            #         backbone_only=False,
            #         chain_ids=self.selected_sampler.chain_idx,
            #     )
            #
            #     out = f"{traj_prefix}_pX0_traj.pdb"
            #     writepdb_multi(
            #         out,
            #         px0_xyz_stack,
            #         bfacts,
            #         final_seq.squeeze(),
            #         use_hydrogens=False,
            #         backbone_only=False,
            #         chain_ids=self.selected_sampler.chain_idx,
            #     )

            self.log.info(f"Finished design in {(time.time() - start_time) / 60:.2f} minutes")