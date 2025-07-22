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
                                    '2025-07-22_10-11-46_foldflow',
                                    'foldflow_outputs')



    pdb_names = os.listdir(inference_folder)


    for i in range(100, 201, 100):
        target_folder = create_target_folder(inference_folder, f'foldflow_{i}')


        source_folder = os.path.join(inference_folder, f'length_{i}', 'sample')
        for idx, len_file in enumerate(os.listdir(source_folder)):
            source_file= os.path.join(source_folder, len_file)
            shutil.copy(source_file, f'{target_folder}/{idx}.pdb')