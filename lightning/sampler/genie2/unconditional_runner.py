import argparse
from tqdm import tqdm

from lightning.sampler.genie2.unconditional import UnconditionalSampler
from lightning.sampler.genie2.multiprocessor import MultiProcessor
from lightning.model.genie2.lightning_model import genie2_Lightning_Model
from omegaconf import OmegaConf


class UnconditionalRunner(MultiProcessor):
	"""
	A multi-processing runner for unconditional sampling.
	"""

	def create_tasks(self, infer_conf):
		"""
		Define a set of tasks to be distributed across processes.

		Args:
			infer_conf:
				A OmegaConf of parameters.

		Returns:
			tasks:
				A list of tasks to be distributed across processes, where 
				each task is represented as a dictionary of task-specific 
				parameters.
		"""
		params = OmegaConf.to_container(infer_conf, resolve=True)

		# Initialize
		tasks = []

		# Iterate through lengths
		for length in range(
			params['max_length'],
			params['min_length'] - 1,
			-params['length_step']
		):

			# Create task
			tasks.append({
				'length': length
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
			'weights_path',
			'scale', 'output_dir', 'num_samples', 'batch_size'
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
		model = genie2_Lightning_Model.load_from_checkpoint(constants['weights_path']).eval().to(device)

		# Load sampler
		sampler = UnconditionalSampler(model)

		# Iterate through all tasks
		for task in tqdm(tasks, desc=device):

			# Initialize
			num_samples = constants['num_samples']

			# Sample
			while num_samples > 0:

				# Define
				batch_size = min(constants['batch_size'], num_samples)

				# Initialize parameters
				params = {
					'length': task['length'],
					'scale': constants['scale'],
					'num_samples': batch_size,
					'output_dir': constants['output_dir'],
					'prefix': str(task['length']),
					'offset': constants['num_samples'] - num_samples
				}

				# Sample
				sampler.sample(params)

				# Update
				num_samples -= batch_size


def main(args):

	# Define multiprocessing runner
	runner = UnconditionalRunner()
	
	# Run
	runner.run(vars(args), args.num_devices, args.sequential_order)
		

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
	parser.add_argument('--num_samples', type=int, help='Number of samples per length', default=5)
	parser.add_argument('--batch_size', type=int, help='Batch size', default=4)
	parser.add_argument('--min_length', type=int, help='Minimum sequence length', default=50)
	parser.add_argument('--max_length', type=int, help='Maximum sequence length', default=256)
	parser.add_argument('--length_step', type=int, help='Length step size', default=1)
	
	# Define environment arguments
	parser.add_argument('--num_devices', type=int, help='Number of GPU devices', default=1)
	parser.add_argument('--sequential_order', action='store_true', help='Run in increasing order of length')

	# Parse arguments
	args = parser.parse_args()

	# Run
	main(args)