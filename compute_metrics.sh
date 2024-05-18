#!/bin/bash
sparsity=("0.1")
# data_choice=("aircraft" "cars" "cub" "cifar100" "pets" "gtsrb")
data_choice=("long_seq_classes")

jis=29079022
jie=29079024

for a in "${sparsity[@]}"; do
    for b in "${data_choice[@]}"; do
      # Build the command and submit the job
      command="python compute_metrics.py -jis $jis -jie $jie -dc $b -sp $a"
      echo "Submitting job: $command"
      # Uncomment the line below to actually submit the job
      eval "$command"
    done
  done