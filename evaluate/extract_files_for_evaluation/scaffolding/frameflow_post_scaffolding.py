import os
import shutil



def create_target_folder(inference_folder, target_folder_name):
    target_folder = os.path.join(inference_folder, target_folder_name)
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    return target_folder

if __name__ == '__main__':

    hydra_folder = '../../../lightning_protein/hydra_inference'
    inference_folder = os.path.join(hydra_folder,
                                    '2025-07-22_09-20-57_frameflow',
                                    'frameflow_outputs')
    pdb_names = os.listdir(inference_folder)

    masks_target_folder = create_target_folder(inference_folder, 'masks')
    pdbs_target_folder = create_target_folder(inference_folder,  'pdbs')

    for pdb_name in pdb_names:
        for sample_name in os.listdir(os.path.join(inference_folder, pdb_name)):
            s_id = sample_name.split('.')[-1]

            source_folder = os.path.join(inference_folder, pdb_name, sample_name)
            shutil.copy(f'{source_folder}/sample.pdb', f'{pdbs_target_folder}/{pdb_name}_{s_id}.pdb')
            shutil.copy(f'{source_folder}/motif_mask.npy', f'{masks_target_folder}/{pdb_name}_{s_id}.npy')


