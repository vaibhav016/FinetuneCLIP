#!/bin/bash
seeds=(0 42 3407 1234 101)
data_choice=("aircraft" "cifar100")
########## Aircraft and Cifar with ema and TTA phase=base #######

script_file="individual_jobs.sh True base True False"
for seed in "${seeds[@]}"; do
    for dc in "${data_choice[@]}"; do
      # Build the command and submit the job
      command="sbatch $script_file $dc $seed" 
      echo "Submitting job: $command"
      # Uncomment the line below to actually submit the job
      eval "$command"
    done
  done

########## Aircraft and Cifar with ema and TTA phase=teacher-student #######
script_file="individual_jobs.sh True teacher_student True False"
for seed in "${seeds[@]}"; do
    for dc in "${data_choice[@]}"; do
      # Build the command and submit the job
      command="sbatch $script_file $dc $seed" 
      echo "Submitting job: $command"
      # Uncomment the line below to actually submit the job
      eval "$command"
    done
  done 

######### Aircraft and Cifar with ema and no TTA  #######
script_file="individual_jobs.sh True base False False"
for seed in "${seeds[@]}"; do
    for dc in "${data_choice[@]}"; do
      # Build the command and submit the job
      command="sbatch $script_file $dc $seed" 
      echo "Submitting job: $command"
      # Uncomment the line below to actually submit the job
      eval "$command"
    done
  done 

# ########## Aircraft and Cifar with ema and TTA phase=teacher_student but oracle to measure the upper limit of TTA #######
script_file="individual_jobs.sh True teacher_student True True"
for seed in "${seeds[@]}"; do
    for dc in "${data_choice[@]}"; do
      # Build the command and submit the job
      command="sbatch $script_file $dc $seed" 
      echo "Submitting job: $command"
      # Uncomment the line below to actually submit the job
      eval "$command"
    done
  done 

########## Aircraft and Cifar with ema and TTA phase=base but oracle to measure the upper limit of TTA #######
script_file="individual_jobs.sh True base True True"
for seed in "${seeds[@]}"; do
    for dc in "${data_choice[@]}"; do
      # Build the command and submit the job
      command="sbatch $script_file $dc $seed" 
      echo "Submitting job: $command"
      # Uncomment the line below to actually submit the job
      eval "$command"
    done
  done 



########## Aircraft and Cifar without ema and without TTA #######
script_file="individual_jobs.sh False teacher_student False False"
for seed in "${seeds[@]}"; do
    for dc in "${data_choice[@]}"; do
      # Build the command and submit the job
      command="sbatch $script_file $dc $seed" 
      echo "Submitting job: $command"
      # Uncomment the line below to actually submit the job
      eval "$command"
    done
  done 

