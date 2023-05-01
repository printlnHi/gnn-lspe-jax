#!/bin/bash

run_zinc_with_seeds() {
  seeds=(0 1 2 3)
  program="main_zinc.py"
  for seed in "${seeds[@]}"; do
    command="python ${program} --seed=${seed} $@"
    echo "running" "$command"
    python "${program}" --seed="${seed}" "$@"
  done
}

run_moltox21_with_seeds() {
  seeds=(0 1 2 3)
  program="main_moltox21.py"
  for seed in "${seeds[@]}"; do
    command="python ${program} --seed=${seed} $@"
    echo "running" "$command"
    python "${program}" --seed="${seed}" "$@"
  done
}