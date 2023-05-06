#!/bin/bash

# Run an experiment on the MOLTox21 task across all Positional Embeddings
# Calling convention: moltox21_experiment.sh repo_dir [--wandb_run_name_suffix string] [--wandb] [--min_seed=0] [--max_seed=3] [--|--verbatim_args <args>*]
# Args after -- or --verbatim_args are passed verbatim to main_moltox21.py

set -e # Exit on single error

repo_dir=$1
shift

cd "${repo_dir}"
source scripts/run_functions.sh

fixed_wandb_run_names=(
  "NoPE"
  "LapPE"
  "LSPE"
)

fixed_args=(
  "--config configs/GatedGCN_MOLTOX21_NoPE.json"
  "--config configs/GatedGCN_MOLTOX21_LapPE.json"
  "--config configs/GatedGCN_MOLTOX21_LSPE.json"
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
  moltox21_multi_run "$min_seed" "$max_seed" "$arg"
done