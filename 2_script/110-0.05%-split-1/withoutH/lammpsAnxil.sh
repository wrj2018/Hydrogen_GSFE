#!/bin/bash
#SBATCH --partition=<GPU_PARTITION>      
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=9     
#SBATCH --gres=gpu:1
#SBATCH --output=min%j.out
#SBATCH --error=SFE%j.err

export TMPDIR=<USER_PATH>/tmp
mkdir -p $TMPDIR

module load gcc/14.2.0
# module swap openmpi4 mpi/openmpi/4.1.1_gnu
module load cuda/11.8
module load anaconda/3-2023.09

export LD_LIBRARY_PATH=<USER_PATH>/deepmd-kit/lib:$LD_LIBRARY_PATH
export deepmd_path=<USER_PATH>/deepmd-kit
source activate $deepmd_path

mpirun -np 1 $deepmd_path/bin/lmp_mpi -in gsfe.in