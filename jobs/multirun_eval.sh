#!/bin/bash           

#SBATCH --job-name=multirun_eval_mos
#SBATCH --partition=svl
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=20G
#SBATCH --gres=gpu:1
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=na

source ~/.bashrc  
conda activate mos

python scripts/run.py --multirun +eval=predict_location ++changes_per_step=0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 ++observation_prob=0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,1.0
