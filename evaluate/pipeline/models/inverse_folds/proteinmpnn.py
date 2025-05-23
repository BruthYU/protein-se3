import os
import copy
import torch
import numpy as np

from evaluate.pipeline.models.inverse_folds.base import InverseFoldModel

import sys
sys.path.append('packages')

from evaluate.ProteinMPNN.protein_mpnn_utils import (
	_scores,
	_S_to_seq,
	tied_featurize,
	parse_PDB
)
from evaluate.ProteinMPNN.protein_mpnn_utils import ProteinMPNN as ProteinMPNNBase
from evaluate.ProteinMPNN.protein_mpnn_utils import StructureDatasetPDB
CA_ONLY = True

class ProteinMPNN(InverseFoldModel):
	"""
	ProteinMPNN inverse folding model, adapted from ProteinMPNN Github repository 
	(https://github.com/dauparas/ProteinMPNN).

	NOTE: We use the default setting from ProteinMPNN Github repository and sequences are 
	predicted based on the Ca atom coordinates from the input PDB-formatted structure.
	"""

	def __init__(
		self,
		rootdir=os.path.join('../..', 'ProteinMPNN'),
		model_name='v_48_020',
		device='cuda:0',
		num_samples=16,
		sampling_temperature=0.1,
	):
		"""
		Args:
			rootdir:
				Root directory for ProteinMPNN package. Default to 'packages/ProteinMPNN'.
			model_name:
				Model name for ProteinMPNN. Default to 'v_48_020'.
			device:
				Device name. Default to 'cuda:0'.
			num_samples:
				Number of samples to be drawn from the sequence distribution. Default to 8.
			sampling_temperature:
				Temperature in sampling process. Default to 0.1.
		"""
		
		# Load checkpoint
		if CA_ONLY:
			checkpoint_path = os.path.join(rootdir, 'ca_model_weights', f'{model_name}.pt')
		else:
			checkpoint_path = os.path.join(rootdir, 'vanilla_model_weights', f'{model_name}.pt')
		checkpoint = torch.load(checkpoint_path)

		# Load model
		self.model = ProteinMPNNBase(
			ca_only=CA_ONLY,
			num_letters=21,
			node_features=128,
			edge_features=128,
			hidden_dim=128,
			num_encoder_layers=3,
			num_decoder_layers=3,
			augment_eps=0,
			k_neighbors=checkpoint['num_edges']
		)
		self.model.name = model_name
		self.model.device = device
		self.model.load_state_dict(checkpoint['model_state_dict'])
		self.model.eval().to(device)

		# Save sampling parameters
		self.num_samples = num_samples
		self.sampling_temperature = sampling_temperature

	def predict(
		self,
		pdb_filepath,
		designed_chain_list=['A'],
		fixed_chain_list=[],
		fixed_positions_dict=None
	):
		"""
		Predict sequences given an input pdb filepath.

		Args:
			pdb_filepath:
				PDB filepath of input structure.
			designed_chain_list:
				List of chains to be designed. Default to ['A'].
			fixed_chain_list:
				List of chains to be fixed. Default to an empty list.
			fixed_positions_dict:
				Dictionary for specifying fixed residues. For example, given a structure stored
				in example.pdb and fixed residues (13 - 15) from chain A, the fixed positions
				dictionary dictionary is formmated as {"example": {"A": [13, 14, 15]}}. Default
				to None.

		Returns:
			lines:
				Predicted sequences with statistics in FASTA format.
		"""

		chain_list = list(set(designed_chain_list + fixed_chain_list))

		# validate chain list
		with open(pdb_filepath) as file:
			valid_chain_set = set([
				line[21] for line in file
				if line.startswith('ATOM')
			])
		assert set(chain_list) == valid_chain_set, f'Invalid pdb file for ProteinMPNN: {pdb_filepath}'

		#@markdown ### Design Options
		num_seqs = self.num_samples
		num_seq_per_target = num_seqs

		#@markdown - Sampling temperature for amino acids, T=0.0 means taking argmax, T>>1.0 means sample randomly.
		sampling_temp = str(self.sampling_temperature)

		##############################################################

		save_score=0                      # 0 for False, 1 for True; save score=-log_prob to npy files
		save_probs=0                      # 0 for False, 1 for True; save MPNN predicted probabilites per position
		score_only=0                      # 0 for False, 1 for True; score input backbone-sequence pairs
		conditional_probs_only=0          # 0 for False, 1 for True; output conditional probabilities p(s_i given the rest of the sequence and backbone)
		conditional_probs_only_backbone=0 # 0 for False, 1 for True; if true output conditional probabilities p(s_i given backbone)

		batch_size=1                      # Batch size; can set higher for titan, quadro GPUs, reduce this if running out of GPU memory
		max_length=20000                  # Max sequence length

		out_folder='.'                    # Path to a folder to output sequences, e.g. /home/out/
		jsonl_path=''                     # Path to a folder with parsed pdb into jsonl
		omit_AAs='X'                      # Specify which amino acids should be omitted in the generated sequence, e.g. 'AC' would omit alanine and cystine.

		pssm_multi=0.0                    # A value between [0.0, 1.0], 0.0 means do not use pssm, 1.0 ignore MPNN predictions
		pssm_threshold=0.0                # A value between -inf + inf to restric per position AAs
		pssm_log_odds_flag=0               # 0 for False, 1 for True
		pssm_bias_flag=0                   # 0 for False, 1 for True

		##############################################################

		folder_for_outputs = out_folder

		NUM_BATCHES = num_seq_per_target//batch_size
		BATCH_COPIES = batch_size
		temperatures = [float(item) for item in sampling_temp.split()]
		omit_AAs_list = omit_AAs
		alphabet = 'ACDEFGHIKLMNPQRSTVWYX'

		omit_AAs_np = np.array([AA in omit_AAs_list for AA in alphabet]).astype(np.float32)

		chain_id_dict = None
		fixed_positions_dict = fixed_positions_dict
		pssm_dict = None
		omit_AA_dict = None
		bias_AA_dict = None
		tied_positions_dict = None
		bias_by_res_dict = None
		bias_AAs_np = np.zeros(len(alphabet))

		###############################################################

		pdb_dict_list = parse_PDB(pdb_filepath, input_chain_list=chain_list)
		dataset_valid = StructureDatasetPDB(pdb_dict_list, truncate=None, max_length=max_length)

		chain_id_dict = {}
		chain_id_dict[pdb_dict_list[0]['name']]= (designed_chain_list, fixed_chain_list)

		# print(chain_id_dict)
		for chain in chain_list:
		  l = len(pdb_dict_list[0][f"seq_chain_{chain}"])
		  # print(f"Length of chain {chain} is {l}")

		tied_positions_dict = None

		###############################################################

		lines = ''

		with torch.no_grad():
		  # print('Generating sequences...')
		  for ix, protein in enumerate(dataset_valid):
		    score_list = []
		    all_probs_list = []
		    all_log_probs_list = []
		    S_sample_list = []
		    batch_clones = [copy.deepcopy(protein) for i in range(BATCH_COPIES)]
		    X, S, mask, lengths, chain_M, chain_encoding_all, chain_list_list, \
		    	visible_list_list, masked_list_list, masked_chain_length_list_list, \
		    	chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, tied_pos_list_of_lists_list, \
		    	pssm_coef, pssm_bias, pssm_log_odds_all, bias_by_res_all, tied_beta = tied_featurize(
		    		batch_clones, self.model.device, chain_id_dict, fixed_positions_dict, omit_AA_dict, \
		    		tied_positions_dict, pssm_dict, bias_by_res_dict, ca_only=CA_ONLY
		    	)
		    pssm_log_odds_mask = (pssm_log_odds_all > pssm_threshold).float() #1.0 for true, 0.0 for false
		    name_ = batch_clones[0]['name']

		    randn_1 = torch.randn(chain_M.shape, device=X.device)
		    log_probs = self.model(X, S, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, randn_1)
		    mask_for_loss = mask*chain_M*chain_M_pos
		    scores = _scores(S, log_probs, mask_for_loss)
		    native_score = scores.cpu().data.numpy()

		    for temp in temperatures:
		        for j in range(NUM_BATCHES):
		            randn_2 = torch.randn(chain_M.shape, device=X.device)
		            sample_dict = self.model.sample(X, randn_2, S, chain_M, chain_encoding_all, residue_idx, mask=mask, temperature=temp, omit_AAs_np=omit_AAs_np, bias_AAs_np=bias_AAs_np, chain_M_pos=chain_M_pos, omit_AA_mask=omit_AA_mask, pssm_coef=pssm_coef, pssm_bias=pssm_bias, pssm_multi=pssm_multi, pssm_log_odds_flag=bool(pssm_log_odds_flag), pssm_log_odds_mask=pssm_log_odds_mask, pssm_bias_flag=bool(pssm_bias_flag), bias_by_res=bias_by_res_all)
		            S_sample = sample_dict["S"]
		            log_probs = self.model(X, S_sample, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, randn_2, use_input_decoding_order=True, decoding_order=sample_dict["decoding_order"])
		            mask_for_loss = mask*chain_M*chain_M_pos
		            scores = _scores(S_sample, log_probs, mask_for_loss)
		            scores = scores.cpu().data.numpy()
		            all_probs_list.append(sample_dict["probs"].cpu().data.numpy())
		            all_log_probs_list.append(log_probs.cpu().data.numpy())
		            S_sample_list.append(S_sample.cpu().data.numpy())
		            for b_ix in range(BATCH_COPIES):
		                masked_chain_length_list = masked_chain_length_list_list[b_ix]
		                masked_list = masked_list_list[b_ix]
		                seq_recovery_rate = torch.sum(torch.sum(torch.nn.functional.one_hot(S[b_ix], 21)*torch.nn.functional.one_hot(S_sample[b_ix], 21),axis=-1)*mask_for_loss[b_ix])/torch.sum(mask_for_loss[b_ix])
		                seq = _S_to_seq(S_sample[b_ix], chain_M[b_ix])
		                score = scores[b_ix]
		                score_list.append(score)
		                native_seq = _S_to_seq(S[b_ix], chain_M[b_ix])

		                start = 0
		                end = 0
		                list_of_AAs = []
		                for mask_l in masked_chain_length_list:
		                    end += mask_l
		                    list_of_AAs.append(seq[start:end])
		                    start = end

		                seq = "".join(list(np.array(list_of_AAs)[np.argsort(masked_list)]))
		                l0 = 0
		                for mc_length in list(np.array(masked_chain_length_list)[np.argsort(masked_list)])[:-1]:
		                    l0 += mc_length
		                    seq = seq[:l0] + '/' + seq[l0:]
		                    l0 += 1
		                score_print = np.format_float_positional(np.float32(score), unique=False, precision=4)
		                seq_rec_print = np.format_float_positional(np.float32(seq_recovery_rate.detach().cpu().numpy()), unique=False, precision=4)
		                line = '>T={}, sample={}, score={}, seq_recovery={}\n{}\n'.format(
		                	temp, b_ix, score_print, seq_rec_print, seq
		                )
		                lines += line

		return lines