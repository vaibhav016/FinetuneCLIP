#!/bin/bash

# ########## CL and without TTA #######
sbatch online_job.sh aircraft True teacher_student False False 0
sbatch online_job.sh cifar100 True teacher_student False False 0
sbatch online_job.sh cars True teacher_student False False 0
sbatch online_job.sh cub True teacher_student False False 0
sbatch online_job.sh pets True teacher_student False False 0
sbatch online_job.sh gtsrb True teacher_student False False 0

# ########## MST #######
sbatch online_job.sh aircraft True teacher_student True False 0
sbatch online_job.sh cifar100 True teacher_student True False 0
sbatch online_job.sh cars True teacher_student True False 0
sbatch online_job.sh cub True teacher_student True False 0
sbatch online_job.sh pets True teacher_student True False 0
sbatch online_job.sh gtsrb True teacher_student True False 0


# dc=$1
# em=$2
# tlm=$3
# tp=$4
# or=$5
# s=$6
