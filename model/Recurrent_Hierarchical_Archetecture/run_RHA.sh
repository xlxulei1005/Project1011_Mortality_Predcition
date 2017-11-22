#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=16:59:00
#SBATCH --mem=20GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=rha


module purge
module load h5py/intel/2.7.0rc2
module load cuda/8.0.44
module load pytorch/0.2.0_1
module load torchvision/0.1.8
module load anaconda3/4.3.1
module load scikit-learn/intel/0.18.1
cd ~/mortality_auc/
python training_GPU.py
