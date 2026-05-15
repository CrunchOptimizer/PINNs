#!/bin/bash
#SBATCH -J Burgers_1D
#SBATCH -p gpu-he
#SBATCH -N 1
#SBATCH -n 8
##SBATCH --nodelist=gpu4003
##SBATCH --ntasks-per-node=1
#SBATCH --constraint=nomig
#SBATCH --time=02:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH -o %x_%j.o
#SBATCH -e %x_%j.e

module load python/3.11.11-5e66
source /oscar/data/gk/kshukla1/Optmizer_Leg_2/BURGERS_1D/Case_1_SSBroyden_Zoom/jax_venv/bin/activate

python3 -u Euler_Sod_Linear.py
