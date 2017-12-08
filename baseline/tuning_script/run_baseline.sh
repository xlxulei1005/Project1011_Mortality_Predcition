#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=20:00:00
#SBATCH --mem=15GB
#SBATCH --job-name=tune_baseline
#SBATCH --mail-type=END
#SBATCH --mail-user=lx557@nyu.edu
#SBATCH --output=slurm_%j.out
module purge 
module load scikit-learn/intel/0.18.1
time python train_val.py --epoch_num '10 20 30 40 50' --window_size '2 4 6 8 12 18 24 30' --time_period '15m 6h 12h 24h' --dim_list '150 200 250 300 350 400' --learning_rate_li '0.001 0.005 0.01 0.02 0.05'
