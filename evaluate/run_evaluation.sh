# 1. 把需要评估的Predicted Protein放在Workspace/pdbs文件夹下
# 2. 按顺序执行即可完整评估Designability, Diversity, Novelty



# Quality (scTM scRMSD)
cd pipeline/standard/
python standard_evaluate.py --version unconditional --rootdir ../../workspace


# Diversity Choice
# Pair Wise TM -> tm_info.csv
cd ../diversity
python diversity_evaluate.py --num_cpus 8 --rootdir ../../workspace


cd ../../workspace
# Max cluster (FoldSeek)
#foldseek easy-cluster designs res tmp -c 0.9





# Novelty (FoldSeek)
# if the database has been downloaded already, comment following lines
#mkdir fs_db
cd fs_db
#foldseek databases PDB pdb tmp

foldseek easy-search ../designs pdb ../aln.m8 ../tmpSearchFolder --format-output "query,target,alntmscore" --exhaustive-search --max-accept 5 --num-iterations 3


cd ../..
python final_result.py