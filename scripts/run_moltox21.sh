#!/bin/bash
# Bash script to run all of the moltox21 experiments by calling main_moltox21.py
# Calling convention: bash run_moltox21.sh repo_dir [--wandb_run_name_suffix string] [--wandb] [--|--verbatim_args <args>*]
set -e # Exit on single error

repo_dir=$1
shift

cd "${repo_dir}"
program="main_moltox21.py"
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

seeds=(0 1 2 3)

wandb_run_name_suffix=""
connect_to_wandb=0

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
    --|--verbatim_args)
      shift
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
  arg=" $verbatim ${fixed_args[i]}"
  if [ $connect_to_wandb -eq 1 ]; then
    arg="$arg --wandb_run_name=${fixed_wandb_run_names[i]}$wandb_run_name_suffix --wandb"
  fi
  echo "Running ${program} $arg with seeds ${seeds[@]}"
  for seed in "${seeds[@]}"; do
    echo "Running ${program} $arg with seed ${seed}"
    python $program --seed=${seed} $arg
  done
done