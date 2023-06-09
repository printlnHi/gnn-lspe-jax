#!/bin/bash
echo "Downloading datasets in non MoleculeJraphDataset format"
echo ""
# download mutag. URL from https://github.com/deepmind/educational/blob/master/colabs/summer_schools/intro_to_graph_nets_tutorial_with_jraph.ipynb
wget -P data/molecules https://storage.googleapis.com/dm-educational/assets/graph-nets/jraph_datasets/mutag.pickle
# download ZINC. URL - from https://github.com/vijaydwivedi75/gnn-lspe/blob/main/data/script_download_ZINC.sh
wget -P data/molecules https://data.dgl.ai/dataset/benchmarking-gnns/ZINC.pkl


DIR="$(dirname "$(realpath "$0")")"

echo "Converting datasets to MoleculeJraphDataset format"
python $DIR/../datasets.py 