#!/bin/bash
#SBATCH --job-name=read_h5ad
#SBATCH --partition=unlimited
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=500G
#SBATCH --time=12:00:00

source .venv/bin/activate

python xatlas_align.py
