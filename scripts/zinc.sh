# Bash script to run all of the zinc experiments by calling main_zinc.py
# Calling convention: bash zinc.sh <wandb_run_name_suffix> <connect_to_wandb> 

wandb_run_name_suffix=$1
connect_to_wandb=$2

program="main_zinc.py"

arguments=(
  "--config configs/GatedGCN_ZINC_NoPE.json  --truncate_to=6 --batch_size=2"
  "--config configs/GatedGCN_ZINC_NoPE.json  --truncate_to=20 --batch_size=5"
  "--config configs/GatedGCN_ZINC_NoPE.json  --truncate_to=20 --batch_size=6"
)

wandb_run_names=(
  "proto_zinc_trunc_6_bs2"
  "proto_zinc_trunc_20_bs5"
  "proto_zinc_trunc_20_bs6"
)

for i in "${!arguments[@]}"; do
  echo "Running ${program} ${arguments[i]}"
  if [ $connect_to_wandb = "True" ]; then
    python $program ${arguments[i]} --wandb_run_name=${wandb_run_names[i]}$wandb_run_name_suffix --wandb
  else
    python $program ${arguments[i]}
  fi
done