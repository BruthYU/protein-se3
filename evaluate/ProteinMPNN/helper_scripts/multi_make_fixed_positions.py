import argparse
import os

def main(args):
    import glob
    import random
    import numpy as np
    import json
    import itertools

    with open(args.input_path, 'r') as json_file:
        json_list = list(json_file)

    my_dict = {}
    for pdb_idx, json_str in enumerate(json_list):
        result = json.loads(json_str)
        pure_name = result['name']
        motif_mask_path = os.path.join(args.mask_folder,f'{pure_name}.npy')
        motif_mask = np.load(motif_mask_path)
        fixed_positions = (np.where(motif_mask)[0] + 1).tolist()
        all_chain_list = [item[-1:] for item in list(result) if item[:9] == 'seq_chain']
        fixed_position_dict = {}
        for i, chain in enumerate(all_chain_list):
            fixed_position_dict[chain] = fixed_positions

        my_dict[result['name']] = fixed_position_dict


    with open(args.output_path, 'w') as f:
        f.write(json.dumps(my_dict) + '\n')

    # e.g. output
    # {"5TTA": {"A": [1, 2, 3, 7, 8, 9, 22, 25, 33], "B": []}, "3LIS": {"A": [], "B": []}}


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--input_path", type=str, help="Path to the parsed PDBs")
    argparser.add_argument("--mask_folder", type=str, help="Path to the parsed masks")
    argparser.add_argument("--output_path", type=str, help="Path to the output path")



    args = argparser.parse_args()
    main(args)

