#!/bin/bash
#SBATCH --job-name=mswml                                                                                         # Job name          
#SBATCH --nodes=1                                                                                                # Run all processes on a single node   
#SBATCH --ntasks=1                                                                                               # Run on a single CPU
#SBATCH --mem=40G                                                                                                # Total RAM to be used
#SBATCH --cpus-per-task=64                                                                                       # Number of CPU cores
#SBATCH --gres=gpu:1                                                                                             # Number of GPUs (per node)
#SBATCH -p long                                                                                                  # Use the gpu partition
#SBATCH --time=15:00:00                                                                                          # Specify the time needed for your experiment
#SBATCH --output=/l/users/umaima.rahman/research/sem3/shifts/outputs/output_train/unetr%J.out                       # Standard output log
for real_batch_size in 4
do
	python train.py \
    --seed "1" \
    --path_train_data "/l/users/zijian.li/data/train/flair/" \
    --path_train_gts "/l/users/zijian.li/data/train/gt/" \
    --path_val_data "/l/users/zijian.li/data/dev_in/flair/" \
    --path_val_gts "/l/users/zijian.li/data/dev_in/gt/" \
    --real_batch_size "${real_batch_size}" \
    --path_save "result/merge_seed1_${real_batch_size}"
done
