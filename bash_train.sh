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
real_batch_size=4
for seed in 1
do
	python my_real2.py \
	--seed $seed \
    --path_train_data "/l/users/zijian.li/data/train/flair/" \
    --path_train_gts "/l/users/zijian.li/data/train/gt/" \
    --path_val_data "/l/users/zijian.li/data/dev_in/flair/" \
    --path_val_gts "/l/users/zijian.li/data/dev_in/gt/" \
    --path_tgt_data "/l/users/zijian.li/data/dev_out/flair" \
    --path_tgt_gts "/l/users/zijian.li/data/dev_out/gt" \
    --path_save "result/merge_seed${seed}_batch_size_${real_batch_size}" \
    --real_batch_size "${real_batch_size}"
done
