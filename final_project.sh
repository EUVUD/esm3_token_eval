#!/bin/bash

#SBATCH -J final_project_job

#SBATCH -t 48:00:00

#SBATCH --partition=dept_gpu

#SBATCH --gres=gpu:1

#SBATCH -C A4000

#SBATCH --mem=64G

source ~/.bashrc

conda activate esm3

python3 parallel_esm3_eval.py