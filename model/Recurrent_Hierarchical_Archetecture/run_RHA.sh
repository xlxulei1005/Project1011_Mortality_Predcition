#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=167:59:00
#SBATCH --mem=50GB
#SBATCH --gres=gpu:2
#SBATCH --job-name=rha


module purge
module load h5py/intel/2.7.0rc2
#module load tensorflow/python3.5
module load cuda/8.0.44
#module load torch/intel/20170104
#module load python3/intel/3.6.3
module load pytorch/0.2.0_1
module load torchvision/0.1.8
module load anaconda3/4.3.1
module load scikit-learn/intel/0.18.1
#module load pytorch/python3.5/0.2.0_3
cd ~/mortality_auc/
python training_GPU.py
