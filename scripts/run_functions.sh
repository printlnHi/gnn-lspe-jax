#!/bin/bash

run_for_seeds(){
  min_seed=$1
  max_seed=$2
  base_command=$3
  seeds=$(seq $min_seed $max_seed)
  echo "seeds:" $seeds
  echo "base command:" $base_command
  for seed in ${seeds[@]}; do 
    command="$base_command --seed=${seed}" 
    echo "running" "$command"
    eval "$command"
  done
}

# Run a zinc experiment with multiple seeds
zinc_multi_run(){
  command="python main_zinc.py ${@:3}"
  run_for_seeds "$1" "$2" "$command"
}

# Run a MolTox21 experiment with multiple seeds
moltox21_multi_run() {
  command="python main_moltox21.py ${@:3}"
  run_for_seeds $1 $2 "$command"
}