#!/bin/bash

#SBATCH -J test_job

#SBATCH -t 48:00:00

#SBATCH --partition=dept_gpu

#SBATCH --gres=gpu:1

#SBATCH --mem=64G

source ~/.bashrc

conda activate esm3

python3 esm3_tok_eval.py