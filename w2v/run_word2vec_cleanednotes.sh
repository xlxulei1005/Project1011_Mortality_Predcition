#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --cpus-per-task=3
#SBATCH --time=25:00:00
#SBATCH --mem=80GB
#SBATCH --job-name=word2vec_cleaned_notes
#SBATCH --mail-type=END
#SBATCH --mail-user=lx557@nyu.edu
#SBATCH --output=slurm_%j.out
module purge
module load nltk/python3.5/3.2.4
module load gensim/intel/python3.5/1.0.1
module load python3/intel/3.5.3
time python3 w2v_preparation_cleaned_notes.py --embedding_dim 300 --mode 1
