#!/bin/bash
#SBATCH --job-name="TTL_CLIP"
#SBATCH --nodes=1
#SBATCH --mem=40G
#SBATCH --gpus-per-node=a100l:1
#SBATCH --time=5:00:00 
#SBATCH --output=/network/scratch/v/vaibhav.singh/slurm-%j.out
#SBATCH --error=/network/scratch/v/vaibhav.singh/slurm-%j.err
#SBATCH --cpus-per-task=1


module restore
module load anaconda/3
source activate llm_project_2
module load cuda/12.1.1 
mkdir $SLURM_TMPDIR/output


cp ../aircrafts_data.zip $SLURM_TMPDIR
cp ../cars.zip $SLURM_TMPDIR
cp ../CUB_200_2011.tgz $SLURM_TMPDIR
# cp ../food-101.tar.gz $SLURM_TMPDIR
# cp ../SUN397.tar.gz $SLURM_TMPDIR
# cp ../cifar-10-python.tar.gz $SLURM_TMPDIR
cp ../cifar-100-python.tar.gz $SLURM_TMPDIR
cp ../GTSRB-Training_fixed.zip $SLURM_TMPDIR
cp ../GTSRB_Final_Test_Images.zip $SLURM_TMPDIR
cp ../GTSRB_Final_Test_GT.zip $SLURM_TMPDIR

cp ../annotations.tar.gz $SLURM_TMPDIR # Pets
cp ../images.tar.gz $SLURM_TMPDIR

unzip -nq $SLURM_TMPDIR/aircrafts_data.zip -d $SLURM_TMPDIR/aircrafts_data
unzip -nq $SLURM_TMPDIR/cars.zip -d $SLURM_TMPDIR/cars
tar -xzf $SLURM_TMPDIR/CUB_200_2011.tgz -C $SLURM_TMPDIR
# tar -xzf $SLURM_TMPDIR/food-101.tar.gz -C $SLURM_TMPDIR
# tar -xvzf $SLURM_TMPDIR/SUN397.tar.gz -C $SLURM_TMPDIR
# tar -xvzf $SLURM_TMPDIR/cifar-10-python.tar.gz -C $SLURM_TMPDIR
tar -xzf $SLURM_TMPDIR/cifar-100-python.tar.gz -C $SLURM_TMPDIR
tar -xzf $SLURM_TMPDIR/annotations.tar.gz -C $SLURM_TMPDIR
tar -xzf $SLURM_TMPDIR/images.tar.gz -C $SLURM_TMPDIR
unzip -nq $SLURM_TMPDIR/GTSRB-Training_fixed.zip -d $SLURM_TMPDIR/gtsrb
unzip -nq $SLURM_TMPDIR/GTSRB_Final_Test_Images.zip -d $SLURM_TMPDIR/gtsrb
unzip -nq $SLURM_TMPDIR/GTSRB_Final_Test_GT.zip -d $SLURM_TMPDIR/gtsrb


dc=$1
mt=$2
s=$3
tp=$4

python mainn.py dataset=$dc method=$mt save_path=$SLURM_TMPDIR/output +finetune_proj=False seed=$s tta_phase=$tp

cp -r $SLURM_TMPDIR/output $SCRATCH 

exit 0

