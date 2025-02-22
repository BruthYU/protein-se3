import os
import glob
import argparse
from tqdm import tqdm
import pandas as pd
from lightning.sampler.genie2.scaffold import ScaffoldSampler
from lightning.sampler.genie2.multiprocessor import MultiProcessor
from lightning.model.genie2.lightning_model import genie2_Lightning_Model
from omegaconf import OmegaConf

class ScaffoldRunner(MultiProcessor):
	"""
	A multi-processing runner for sampling scaffold given motif specifications.
	"""
	def create_tasks(self, infer_conf):
		# Initialize
		tasks = []
		task_df = pd.read_csv(infer_conf.csv_path)
		for idx, row in task_df.iterrows():
			tasks.append({'motif_row': row})
		return tasks



	def old_create_tasks(self, infer_conf):
		"""
		Define a set of tasks to be distributed across processes.

		Args:
			infer_conf:
				A dictionary of parameters.

		Returns:
			tasks:
				A list of tasks to be distributed across processes, where 
				each task is represented as a dictionary of task-specific 
				parameters.
		"""

		# Initialize
		tasks = []

		# Define motif names
		if infer_conf.motif_name is not None:
			motif_names = [infer_conf.motif_name]
		else:
			motif_names = [
				filepath.split('/')[-1].split('.')[0]
				for filepath in glob.glob(os.path.join(infer_conf.datadir, '*.pdb'))
			]

		# Create tasks
		for motif_name in motif_names:
			tasks.append({
				'motif_name': motif_name
			})

		return tasks

	def create_constants(self, infer_conf):
		"""
		Define a dictionary of constants shared across processes.

		Args:
			infer_conf:
				A OmegaConf of parameters.

		Returns:
			constants:
				A dictionary of constants shared across processes.
		"""
		params = OmegaConf.to_container(infer_conf, resolve=True)

		# Define
		names = [
			'weights_path', 'scale', 'strength', 'csv_path',
			'output_dir', 'num_samples', 'batch_size', 'datadir'
		]

		# Create constants
		constants = dict([(name, params[name]) for name in names])

		return constants

	def execute(self, constants, tasks, device):
		"""
		Execute a set of assigned tasks on a given device.

		Args:
			constants:
				A dictionary of constants.
			tasks:
				A list of tasks, where each task is represented as a 
				dictionary of task-specific parameters.
			device:
				Name of device to execute on.
		"""


		# Load model
		# model = load_pretrained_model(
		# 	constants['rootdir'],
		# 	constants['name'],
		# 	constants['epoch']
		# ).eval().to(device)
		model = genie2_Lightning_Model.load_from_checkpoint(constants['weights_path']).eval().to(device)

		# Load sampler
		sampler = ScaffoldSampler(model)

		# Iterate through all tasks
		for task in tqdm(tasks, desc=device):

			# Define output directory
			output_dir = os.path.join(
				constants['output_dir'],
				'motif={}'.format(task['motif_row'].target)
			)

			# Initialize
			num_samples = constants['num_samples']

			# Sample
			while num_samples > 0:

				# Define
				batch_size = min(constants['batch_size'], num_samples)
				filepath = os.path.join(
					constants['datadir'],
					'{}.pdb'.format(task['motif_row'].target)
				)

				# Initialize parameters
				params = {
					'filepath': filepath,
					'motif_row': task['motif_row'],
					'scale': constants['scale'],
					'strength': constants['strength'],
					'num_samples': batch_size,
					'output_dir': output_dir,
					'prefix': task['motif_row'].target,
					'offset': constants['num_samples'] - num_samples
				}

				# Sample
				sampler.sample(params)

				# Update
				num_samples -= batch_size


def main(args):

	# Define multiprocessor runner
	runner = ScaffoldRunner()

	# Run
	runner.run(vars(args), args.num_devices)


if __name__ == '__main__':
	# Create parser
	parser = argparse.ArgumentParser()

	# Define model arguments
	parser.add_argument('--name', type=str, help='Model name', required=True)
	parser.add_argument('--epoch', type=int, help='Model epoch', required=True)
	parser.add_argument('--rootdir', type=str, help='Root directory', default='results')

	# Define sampling arguments
	parser.add_argument('--scale', type=float, help='Sampling noise scale', required=True)
	parser.add_argument('--output_dir', type=str, help='Output directory', required=True)
	parser.add_argument('--strength', type=float, help='Sampling classifier-free strength', default=0)
	parser.add_argument('--num_samples', type=int, help='Number of samples per length', default=1)
	parser.add_argument('--batch_size', type=int, help='Batch size', default=4)
	parser.add_argument('--motif_name', type=str, help='Motif name', default=None)
	parser.add_argument('--datadir', type=str, help='Data directory', default='data/design25')
	
	# Define environment arguments
	parser.add_argument('--num_devices', type=int, help='Number of GPU devices', default=1)

	# Parse arguments
	args = parser.parse_args()

	# Run
	main(args)