#!/bin/bash
#SBATCH -A <ACCOUNT>
#SBATCH --job-name=test
#SBATCH --partition=<GPU_PARTITION>      
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1     
#SBATCH --gres=gpu:4
#SBATCH --output=%j.out
#SBATCH --error=%j.err

export TMPDIR=<USER_PATH>/tmp
mkdir -p $TMPDIR

module load gcc/14.2.0
# module swap openmpi4 mpi/openmpi/4.1.1_gnu
module load cuda/11.8
module load anaconda/3-2023.09

export LD_LIBRARY_PATH=<USER_PATH>/deepmd-kit/lib:$LD_LIBRARY_PATH
export deepmd_path=<USER_PATH>/deepmd-kit
source activate $deepmd_path

mpirun -np 4 $deepmd_path/bin/lmp_mpi -in gsfe.in -var r 1
mpirun -np 4 $deepmd_path/bin/lmp_mpi -in gsfe.in -var r 2
mpirun -np 4 $deepmd_path/bin/lmp_mpi -in gsfe.in -var r 3
mpirun -np 4 $deepmd_path/bin/lmp_mpi -in gsfe.in -var r 4