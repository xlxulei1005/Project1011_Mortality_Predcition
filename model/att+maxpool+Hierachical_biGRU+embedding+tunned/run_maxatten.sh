#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=47:59:00
#SBATCH --mem=25GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=15m_100_maxattn
#SBATCH --mail-type=END
#SBATCH --mail-user=lx557@nyu.edu
module purge
module load h5py/intel/2.7.0rc2
module load cuda/8.0.44
module load pytorch/0.2.0_1
module load torchvision/0.1.8
module load anaconda3/4.3.1
module load scikit-learn/intel/0.18.1
python main.py
