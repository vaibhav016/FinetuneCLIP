#!/bin/bash

module load gcc/9.3.0 arrow/8 python/3.8
source clip_venv/bin/activate
mkdir $SLURM_TMPDIR/output

# cp ../aircrafts_data.zip $SLURM_TMPDIR
# cp ../cars.zip $SLURM_TMPDIR
# cp ../CUB_200_2011.tgz $SLURM_TMPDIR
# cp ../food-101.tar.gz $SLURM_TMPDIR
# # cp ../SUN397.tar.gz $SLURM_TMPDIR
# # cp ../cifar-10-python.tar.gz $SLURM_TMPDIR
# cp ../cifar-100-python.tar.gz $SLURM_TMPDIR
# cp ../GTSRB-Training_fixed.zip $SLURM_TMPDIR
# cp ../GTSRB_Final_Test_Images.zip $SLURM_TMPDIR
# cp ../GTSRB_Final_Test_GT.zip $SLURM_TMPDIR

cp ../annotations.tar.gz $SLURM_TMPDIR # Pets
cp ../images.tar.gz $SLURM_TMPDIR

# unzip -nq $SLURM_TMPDIR/aircrafts_data.zip -d $SLURM_TMPDIR/aircrafts_data
# unzip -nq $SLURM_TMPDIR/cars.zip -d $SLURM_TMPDIR/cars
# tar -xzf $SLURM_TMPDIR/CUB_200_2011.tgz -C $SLURM_TMPDIR
# tar -xzf $SLURM_TMPDIR/food-101.tar.gz -C $SLURM_TMPDIR
# tar -xvzf $SLURM_TMPDIR/SUN397.tar.gz -C $SLURM_TMPDIR
# tar -xvzf $SLURM_TMPDIR/cifar-10-python.tar.gz -C $SLURM_TMPDIR
# tar -xzf $SLURM_TMPDIR/cifar-100-python.tar.gz -C $SLURM_TMPDIR
tar -xzf $SLURM_TMPDIR/annotations.tar.gz -C $SLURM_TMPDIR
tar -xzf $SLURM_TMPDIR/images.tar.gz -C $SLURM_TMPDIR
# unzip -nq $SLURM_TMPDIR/GTSRB-Training_fixed.zip -d $SLURM_TMPDIR/gtsrb
# unzip -nq $SLURM_TMPDIR/GTSRB_Final_Test_Images.zip -d $SLURM_TMPDIR/gtsrb
# unzip -nq $SLURM_TMPDIR/GTSRB_Final_Test_GT.zip -d $SLURM_TMPDIR/gtsrb



ls $SLURM_TMPDIR

# export SSL_CERT_FILE=$(python -m certifi) for solving ssl issue 