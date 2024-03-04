#!/bin/bash

#!/bin/bash

# ########## Aircraft and Cifar without ema and without TTA #######
sbatch individual_jobs.sh aircraft False teacher_student False True 0 
sbatch individual_jobs.sh cifar100 False teacher_student False True 0 

# ######### Aircraft and Cifar with ema and no TTA  #######
sbatch individual_jobs.sh aircraft True base False False 0 
sbatch individual_jobs.sh cifar100 True base False False 0

########## Aircraft and Cifar with ema and TTA phase=base #######
sbatch individual_jobs.sh aircraft True base True False 0
sbatch individual_jobs.sh cifar100 True base True False 0

# ########## Aircraft and Cifar with ema and TTA phase=base but oracle to measure the upper limit of TTA #######
sbatch individual_jobs.sh aircraft True base True True 0
sbatch individual_jobs.sh cifar100 True base True True 0 

########## Aircraft and Cifar with ema and TTA phase=teacher-student #######
sbatch individual_jobs.sh aircraft True teacher_student True False 0
sbatch individual_jobs.sh cifar100 True teacher_student True False 0 

# # ########## Aircraft and Cifar with ema and TTA phase=teacher_student but oracle to measure the upper limit of TTA #######
sbatch individual_jobs.sh aircraft True teacher_student True True 0 
sbatch individual_jobs.sh cifar100 True teacher_student True True 0 






# dc=$1
# em=$2
# tlm=$3
# tp=$4
# or=$5
# s=$6