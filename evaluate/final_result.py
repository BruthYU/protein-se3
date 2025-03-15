import pandas as pd
import protfasta
import os

workspace_dir = 'workspace'
final_result = {}

standard_info_file = pd.read_csv(os.path.join(workspace_dir,'info.csv'))
for col in standard_info_file.columns:
    if col != 'domain':
        final_result[f'avg_{col}'] = standard_info_file[col].mean()


tm_info_file = pd.read_csv(os.path.join(workspace_dir,'tm_info.csv'))

for col in tm_info_file.columns:
    if not col.startswith('Unnamed'):
        final_result[col] = float(tm_info_file[col])

cluster_file = protfasta.read_fasta(os.path.join(workspace_dir, "res_rep_seq.fasta"))
final_result['cluster_num'] = len(cluster_file)
final_result['cluster_percent'] = len(cluster_file) / final_result['protein_num']

novelty_file =  pd.read_csv(os.path.join(workspace_dir, "aln.m8"), sep='\t',header=None)
final_result['avg_novelty_tm'] = novelty_file[novelty_file.columns[-1]].mean()
for key in final_result.keys():
    print(f'{key} : {final_result[key]}')
pass

