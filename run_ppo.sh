#!/bin/bash
#SBATCH --account=jl_615_1279
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-task=p100:1
#SBATCH --mem=20G
#SBATCH --time=24:00:00
#SBATCH --mail-type=all
#SBATCH --mail-user=ylu62702@usc.edu
module purge
eval "$(conda shell.bash hook)"
conda activate autobs

#nvidia-smi

export PYTHONPATH=$PWD:$PYTHONPATH
python runner/run_ppo.py