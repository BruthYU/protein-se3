# Evaluation

## Designablity
Following "[*Out of Many, One: Designing and Scaffolding Proteins at the Scale of the Structural Universe with Genie 2*](https://github.com/aqlaboratory/genie2)", the designablity is evaluated with following steps:
- A structure that can be plausibly realized by some protein sequence is one that is designable. To determine if a structure is designable we employ a commonly used [pipeline](https://github.com/blt2114/ProtDiff_SMCDiff) that computes in silico self-consistency between generated and predicted structures. 

- First, a generated structure is fed into an **inverse folding model** ([ProteinMPNN](https://github.com/dauparas/ProteinMPNN)) to produce 8 plausible sequences for the design. 

- Next, **structures of proposed sequences are predicted** (using [ESMFold](https://github.com/facebookresearch/esm)) and the consistency of predicted structures with respect to the original generated structure is assessed using a structure similarity metric ([TM-score](https://onlinelibrary.wiley.com/doi/10.1002/prot.20264))

In this folder, this metric could be easily computed by running:

```
python design_and_evaluated.py --input_dir <input_dir_path> --output_dir <output_dir_path>
```