#!/bin/bash

#SBATCH -N 1
#SBATCH -o dup_training.log
#SBATCH --partition=GPU
#SBATCH --ntasks-per-node 8
#SBATCH --gres=gpu:8
#SBATCH --time=48:00:00
#SBATCH --dependency=singleton

module load anaconda3
module load cuda
module load cudnn

conda activate /ocean/projects/ele220002p/tongshen/env/3d
srun torchrun --nproc_per_node 8 --nnodes 1 lib/train/run_training.py --script mixformer_vit --config baseline --save_dir work_dirs/mixformer_vit_test --use_lmdb 0 --script_prv None --config_prv baseline  --distill 0 --script_teacher None --config_teacher None --stage1_model None