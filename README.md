# GNN-LSPE-JAX
First and foremost, this is a replication of Dwivedi et al. (2021) (arXiv:2110.07875).
Their original PyTorch codebase can found at https://github.com/vijaydwivedi75/gnn-lspe
The following files are either directories copies or reduced versions rom the original codebase, which is under a MIT license, for the purposes of dataset conversion:
 - data/molecules/*.index
 - data/molecules.py
 - data/ogb_mol.py
The files of configs/ are a slightly modified strict subset of the original config files.

# Replication
## Environment
To allow for easy replication, there are three methods of environment creation: 

Creating a conda environment:
```bash
conda env create -f environment_specs/environment.yml
```
or 
```bash
mamba env create -f environment_specs/environment.yml
```

Installing the required packages via pip (this does not create a new environment):
```bash
pip install -r environment_specs/requirements.txt
```

Installing the required packages via pip if you are on Google Colab (last verified on 6-May-2023):
```bash
pip install -r environment_specs/requirements_colab.txt
```

## Dataset preparation

To download the datasets, and store them as pickled instances of `MoleculeJraphDataset`:
```bash
scripts/download_datasets.sh
python convert_datasets.py
```


## Replication of results from Dwivedi et al. (2021)
Run the `DwivediReplication.ipynb` notebook. This will run the experiments and log the results in Weights and Biases.

## Replication of further experiments 
## Individual runs:
The following three examples are for ZINC but can be trivially adapted to Moltox21
1. To train a single model on the ZINC task:
```bash
python main_zinc.py --config configs/GatedGCN_ZINC_<PE>.json [options]
```

2. To train a model on the ZINC task for seeds 0 through to 9:
```bash
source scripts/run_functions.sh
run_zinc_with_seeds 0 9 --config configs/GatedGCN_ZINC_<PE>.json [options]
```

3. To train a model on the ZINC task for seeds 0 through to 9 for all positional encoding:
```bash
./zinc_experiment.sh --min_seed=0 --max_seed=9 [zinc_experiment.sh options] [-- [options]]
```