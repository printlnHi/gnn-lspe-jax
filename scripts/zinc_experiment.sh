#!/bin/bash

# Run an experiment on the ZINC task across all Positional Embeddings
# Calling convention: zinc_experiment.sh repo_dir [--wandb_run_name_suffix string] [--wandb] [--min_seed=0] [--max_seed=3] [--|--verbatim_args <args>*]
# Args after -- or --verbatim_args are passed verbatim to main_zinc.py
set -e # Exit on single error

repo_dir=$1
if [ -z "${repo_dir}" ] ; then
  echo "Error: no repo dir specified as first argument" >&2
  exit 1
fi
if [ ! -d "${repo_dir}" ] ; then
  echo "Error: ${repo_dir} is not a directory" >&2
  exit 1
fi
cd "${repo_dir}"
shift

source scripts/run_functions.sh

fixed_wandb_run_names=(
  "NoPE"
  "LapPE"
  "LSPE"
  "LapEigLoss"
)

fixed_args=(
  "--config configs/GatedGCN_ZINC_NoPE.json"
  "--config configs/GatedGCN_ZINC_LapPE.json"
  "--config configs/GatedGCN_ZINC_LSPE.json"
  "--config configs/GatedGCN_ZINC_LSPE_withLapEigLoss.json"
)

wandb_run_name_suffix=""
connect_to_wandb=0
min_seed=0
max_seed=3

while true; do
  case $1 in
    --wandb_run_name_suffix)
      wandb_run_name_suffix="_$2"
      shift 2
      ;;
    --wandb)
      connect_to_wandb=1
      shift
      ;;
    --min_seed)
      min_seed=$2
      shift 2
      ;;
    --max_seed)
      max_seed=$2
      shift 2
      ;;
    --|--verbatim_args)
      shift
      break
      ;;
    "")
      break
      ;;
    *)
      echo "Invalid option: $1" >&2
      exit 1
      ;;
  esac
done

verbatim="$@"

for i in "${!fixed_wandb_run_names[@]}"; do
  arg="$verbatim ${fixed_args[i]}"
  if [ $connect_to_wandb -eq 1 ]; then
    arg="$arg --wandb_run_name=${fixed_wandb_run_names[i]}$wandb_run_name_suffix --wandb"
  fi
  echo $i
  zinc_multi_run "$min_seed" "$max_seed" "$arg"
done