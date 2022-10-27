#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=32GB
#SBATCH --time=1:00:00
#SBATCH --partition=gpu 
#SBATCH --gres=gpu:v100:1

module purge
module load gcc/11.3.0
module load python/3.9.12
module load cuda/11.6.2
cd vqa-med-2019/code/MMBERT/vqamed2019/
source ./project/bin/activate

python3 train.py --run_name  "bench_mark1" --mixed_precision --batch_size 16 --num_vis 3 --epochs 50 --hidden_size 768 --num_workers 32
