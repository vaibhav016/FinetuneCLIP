#!/bin/bash

# ########## SPU without TTL on all datasets #######
sbatch mila_job_file.sh cars SPU 1234 False 
sbatch mila_job_file.sh cifar100 SPU 1234 False 
sbatch mila_job_file.sh cub SPU 1234 False 
sbatch mila_job_file.sh pets SPU 1234 False 
sbatch mila_job_file.sh gtsrb SPU 1234 False 


# ########## SPU with TTL on all datasets #######
sbatch mila_job_file.sh cars SPU 1234 True 
sbatch mila_job_file.sh cifar100 SPU 1234 True 
sbatch mila_job_file.sh cub SPU 1234 True 
sbatch mila_job_file.sh pets SPU 1234 True 
sbatch mila_job_file.sh gtsrb SPU 1234 True 

# dc=$1
# mt=$2
# s=$3
# tp=$4

