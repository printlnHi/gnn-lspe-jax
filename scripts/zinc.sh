#!/bin/bash
# Bash script to run all of the zinc experiments by calling main_zinc.py
# Calling convention: bash zinc.sh repo_dir [--wandb_run_name_suffix string] [--wandb] [--min_truncation int] [--max_truncation int] [--|--verbatim_args <args>*]

repo_dir=$1
shift

cd "${repo_dir}"
program="main_zinc.py"

config="configs/GatedGCN_ZINC_NoPE.json"

truncations=(6 6 20 256 1024)

fixed_wandb_run_names=(
  "proto_zinc_trunc_6_bs2"
  "proto_zinc_trunc_6_bs4"
  "proto_zinc_trunc_20_bs5"
  "proto_zinc_trunc_256"
  "proto_zinc_trunc_1024"
)

fixed_args=(
  "--batch_size=2"
  "--batch_size=4"
  "--batch_size=5"
  ""
  ""
)

wandb_run_name_suffix=""
connect_to_wandb=0
min_truncation=0
max_truncation=100000000

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
    --min_truncation)
      min_truncation=$2
      shift 2
      ;;
    --max_truncation)
      max_truncation=$2
      shift 2
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

#Run only programs which meet criteria
for i in "${!fixed_wandb_run_names[@]}"; do
  if [ ${truncations[i]} -ge $min_truncation ] && [ ${truncations[i]} -le $max_truncation ]; then
    arg="--config=$config --truncate_to=${truncations[i]} $verbatim ${fixed_args[i]}"
    if [ $connect_to_wandb -eq 1 ]; then
      arg="$arg --wandb_run_name=${fixed_wandb_run_names[i]}$wandb_run_name_suffix --wandb"
    fi
    echo "Running ${program} $arg"
    python $program $arg
  fi
done