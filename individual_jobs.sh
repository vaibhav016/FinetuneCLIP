#!/bin/bash
#SBATCH --account=def-ravanelm          # Prof Eugene
#SBATCH --cpus-per-task=1               # Ask for 1 CPUs
#SBATCH --gres=gpu:1                     # Ask for 1 GPU
#SBATCH --mem=40G                        # Ask for 32 GB of RAM
#SBATCH --time=1:30:00                   # The job will run for 9 hours
#SBATCH -o /scratch/vs2410/slurm-%j.out  # Write the log in $SCRATCH

module load gcc/9.3.0 arrow/8 python/3.8
source clip_env/bin/activate
mkdir $SLURM_TMPDIR/output

dc=$1
em=$2
tlm=$3
tp=$4
or=$5
s=$6

python mainn.py dataset=$dc method=Finetune save_path=$SLURM_TMPDIR/output +finetune_proj=False buffer_size=0 tta_phase=$tp ema=$em tta_loss_mode=$tlm oracle=$or seed=$s

cp -r $SLURM_TMPDIR/output $SCRATCH 
exit 0

