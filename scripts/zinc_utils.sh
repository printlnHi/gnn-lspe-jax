#!/bin/bash

run_with_seeds() {
  seeds=(0 1 2 3)
  program="main_zinc.py"
  for seed in "${seeds[@]}"; do
    command="python ${program} --seed=${seed} $@"
    echo "running" "$command"
    python "${program}" --seed="${seed}" "$@"
  done
}