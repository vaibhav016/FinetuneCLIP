#!/bin/bash
#SBATCH --account=rrg-eugenium          # Prof Eugene
#SBATCH --cpus-per-task=1               # Ask for 1 CPUs
#SBATCH --gres=gpu:1                     # Ask for 1 GPU
#SBATCH --mem=40G                        # Ask for 32 GB of RAM
#SBATCH --time=1:30:00                   # The job will run for 9 hours
#SBATCH -o /scratch/vs2410/slurm-%j.out  # Write the log in $SCRATCH

module load gcc/9.3.0 arrow/8 python/3.8
source clip_env/bin/activate
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
em=$2
tlm=$3
tp=$4
or=$5
s=$6

python mainn.py dataset=$dc method=Finetune data=$SLURM_TMPDIR save_path=$SLURM_TMPDIR/output +finetune_proj=False buffer_size=0 tta_phase=$tp ema=$em tta_loss_mode=$tlm oracle=$or seed=$s tta_epochs=1 epochs=1

cp -r $SLURM_TMPDIR/output $SCRATCH 

exit 0

