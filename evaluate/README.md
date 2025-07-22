# Evaluation

## Unconditional Generation

- Step 0: Remove old files of history evaluation (if exist): ```designs/,sequences/,scores/,alm.m8,info.csv,tm_info.csv,final_result_un.csv```
- Step 1: Place the predicted proteins to be evaluated in the folder ```evaluate/workspace/pdbs```.
- Step 2: Install [FoldSeek](https://github.com/steineggerlab/foldseek) and download pre-generated databases
```shell
  cd evaluate/workspace
  mkdir fs_db
  cd fs_db
  foldseek databases PDB pdb tmp
```
- Step 3: Run the evaluation shell script 
```shell
  cd evaluate/
  sh run_evaluation_un.sh
```
The evaluation results will be written into ```evaluate/workspace/final_result_un.csv``` 