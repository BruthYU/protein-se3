# Evaluation
First, create a folder as the evaluation worksapce:
```shell
  cd evaluate
  mkdir workspace
```
## Unconditional Generation

- Step 0: Remove old files of history evaluation (if exist): ```designs/,sequences/,scores/,alm.m8,info.csv,tm_info.csv,final_result_un.csv,/jsonl```
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


## Motif Scaffolding

- Step 0: Remove old files of history evaluation (if exist): ```designs/,sequences/,scores/,alm.m8,info.csv,tm_info.csv,final_result_un.csv,/jsonl```
- Step 1: Place the predicted proteins to be evaluated in the folder ```evaluate/workspace/pdbs```, and the generated mask files into  ```evaluate/workspace/masks```
- Step 2: Run the evaluation shell script 
```shell
  cd evaluate/
  sh run_evaluation_mo.sh
```
The evaluation results will be written into ```evaluate/workspace/info.csv``` 