# LieOTDocking

This repository contains the code and data for the LieOTDocking project, a framework for protein-protein docking using Lie Group optimization and Optimal Transport.

## Scripts

### `LieOT_v28_hybrid.py`

This script implements a hybrid docking pipeline. It uses MSMS for surface generation and an advanced feature scoring method to identify potential binding sites. These initial poses are then refined to produce a final docked structure.

#### Usage

Here is a basic command to run a docking simulation on the 1AVW test case:

```bash
python LieOT_v28_hybrid.py \
    --ligand 1AVW_B.pdb \
    --receptor 1AVW_A.pdb \
    --reference reference.pdb \
    --num_features 50 \
    --patch_radius 6 \
    --msms_density 0.5 \
    --output docked_1AVW_v28_hybrid.pdb
```

*   `--ligand`: PDB file of the ligand (e.g., `1AVW_B.pdb`).
*   `--receptor`: PDB file of the receptor (e.g., `1AVW_A.pdb`).
*   `--reference`: PDB file of the native complex, used for calculating the RMSD of the output poses (e.g., `reference.pdb`).
*   `--output`: Name for the output file which will contain the best pose found.

The script has other parameters for fine-tuning the docking process, such as `--num_features`, `--patch_radius`, and optimization hyperparameters. You can see all options by running `python LieOT_v28_hybrid.py --help`.
~                                                                                                                                                                                                                               
