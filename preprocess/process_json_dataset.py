"""Script for preprocessing mmcif files for faster consumption.

- Parses all mmcif protein files in a directory.
- Filters out low resolution files.
- Performs any additional processing.
- Writes all processed examples out to specified path.
"""

import argparse
import dataclasses
import functools as fn
import json
import multiprocessing as mp
import os
import time
import sys
import numpy as np
import pandas as pd

sys.path.append('..')
from preprocess.tools import utils as du, errors, parsers



# Define the parser
parser = argparse.ArgumentParser(
    description='mmCIF processing script.')
parser.add_argument(
    '--jsonl_path',
    help='Path to jsonl files.',
    type=str,
    default='./raw/chain_set.jsonl')
parser.add_argument(
    '--max_file_size',
    help='Max file size.',
    type=int,
    default=3000000)  # Only process files up to 3MB large.
parser.add_argument(
    '--min_file_size',
    help='Min file size.',
    type=int,
    default=1000)  # Files must be at least 1KB.
parser.add_argument(
    '--max_resolution',
    help='Max resolution of files.',
    type=float,
    default=5.0)

parser.add_argument(
    '--num_processes',
    help='Number of processes.',
    type=int,
    default=1)
parser.add_argument(
    '--write_dir',
    help='Path to write results to.',
    type=str,
    default='./pkl/jsonl')
parser.add_argument(
    '--debug',
    help='Turn on for debugging.',
    action='store_true')
parser.add_argument(
    '--verbose',
    help='Whether to log everything.',
    action='store_true')


def process_line(line, write_dir: str):
    """Processes a json line into usable, smaller pickles.

    Args:
        jsonl_path: Path to jsonl file to read.
        write_dir: Directory to write pickles to.

    Returns:
        Saves processed protein to pickle and returns metadata.

    Raises:
        DataError if a known filtering rule is hit.
        All other errors are unexpected and are propogated.
    """
    metadata = {}
    line_name = line['name'].lower()

    metadata['pdb_name'] = line_name

    processed_line_path = os.path.join(write_dir, f'{line_name}.pkl')
    # processed_line_path = os.path.abspath(processed_line_path)
    metadata['processed_path'] = processed_line_path





    # Extract all chains
    # chain_id is the last char of the line name
    struct_chains = {line_name.split('.')[-1].upper(): line}
    metadata['num_chains'] = len(struct_chains)

    # Extract features
    struct_feats = []
    all_seqs = set()
    for chain_id, line in struct_chains.items():
        # Convert chain id into int
        chain_id = du.chain_str_to_int(chain_id)
        chain_prot = parsers.process_chain_jsonl(line, chain_id)
        chain_dict = dataclasses.asdict(chain_prot)
        chain_dict = du.parse_chain_feats(chain_dict)
        all_seqs.add(tuple(chain_dict['aatype']))
        struct_feats.append(chain_dict)
    if len(all_seqs) == 1:
        metadata['quaternary_category'] = 'homomer'
    else:
        metadata['quaternary_category'] = 'heteromer'
    complex_feats = du.concat_np_features(struct_feats, False)

    # Process geometry features
    complex_aatype = complex_feats['aatype']
    modeled_idx = np.where(complex_aatype != 20)[0]
    if np.sum(complex_aatype != 20) == 0:
        raise errors.LengthError('No modeled residues')
    min_modeled_idx = np.min(modeled_idx)
    max_modeled_idx = np.max(modeled_idx)
    metadata['seq_len'] = len(complex_aatype)
    metadata['modeled_seq_len'] = max_modeled_idx - min_modeled_idx + 1

    complex_feats['modeled_idx'] = modeled_idx
    # Write features to pickles.
    du.write_pkl(processed_line_path, complex_feats)


    # Return metadata
    return metadata


def process_serially(
        lines, write_dir):
    all_metadata = []
    write_count = 0
    for i, line in enumerate(lines):
        line = json.loads(line)
        line_name = line['name']
        try:
            start_time = time.time()
            metadata = process_line(
                line,
                write_dir)
            write_count = write_count + 1
            elapsed_time = time.time() - start_time
            print(f'Finished {line_name} in {elapsed_time:2.2f}s')
            all_metadata.append(metadata)
        except errors.DataError as e:
            print(f'Failed {line_name}: {e}')
        assert write_count == len(all_metadata), "write_count != len(all_metadata) "
    return all_metadata


def process_fn(
        line,
        verbose=None,
        write_dir=None):
    line = json.loads(line)
    line_name = line['name']
    try:
        start_time = time.time()
        metadata = process_line(
            line,
            write_dir)
        elapsed_time = time.time() - start_time
        if verbose:
            print(f'Finished {line_name} in {elapsed_time:2.2f}s')
        return metadata
    except errors.DataError as e:
        print(f'Failed {line_name}: {e}')


def main(args):
    # Get all mmcif files to read.
    jsonl_path = args.jsonl_path
    with open(jsonl_path) as f:
        lines = f.readlines()
    lines = lines[:1200]
    total_num_paths = len(lines)

    write_dir = args.write_dir
    if not os.path.exists(write_dir):
        os.makedirs(write_dir)
    if args.debug:
        metadata_file_name = 'metadata_debug.csv'
    else:
        metadata_file_name = 'metadata.csv'
    metadata_path = os.path.join(write_dir, metadata_file_name)
    print(f'Files will be written to {write_dir}')

    # Process each mmcif file
    if args.num_processes == 1 or args.debug:
        all_metadata = process_serially(
            lines,
            write_dir)
    else:
        _process_fn = fn.partial(
            process_fn,
            write_dir=write_dir)
        # Uses max number of available cores.
        with mp.Pool() as pool:
            all_metadata = pool.map(_process_fn, lines)
        all_metadata = [x for x in all_metadata if x is not None]
    metadata_df = pd.DataFrame(all_metadata)
    metadata_df.to_csv(metadata_path, index=False)
    succeeded = len(all_metadata)
    print(
        f'Finished processing {succeeded}/{total_num_paths} files')



if __name__ == "__main__":
    # Don't use GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    args = parser.parse_args()
    main(args)