#!/bin/bash

#SBATCH -J dfp
#SBATCH -p gpu
#SBATCH -c 4
#SBATCH -N 1
#SBATCH -t 0-1:30:00
#SBATCH -G 1
#SBATCH --nodelist=yen-gpu3
#SBATCH -o out/%j.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=junze@stanford.edu

# sst - linear layer

CUDA_VISIBLE_DEVICES=0 python3 classifier.py --epochs 20 --fine-tune-mode last-linear-layer --batch_size 16 --lr 1e-3 --hidden_dropout_prob 0.4 --use_gpu --log_file out/cfimdb/training_metrics_batch_16_lr_1e-3_drop_0.4.csv > out/cfimdb/batch_16_lr_1e-3_drop_0.4.log 2>&1 &
echo "started job 2!"

# # cfimdb - full model fine tune
# CUDA_VISIBLE_DEVICES=2 python3 classifier.py --epochs 15 --fine-tune-mode full-model --batch_size 16 --lr 1e-5 --hidden_dropout_prob 0.45 --use_gpu --log_file out/cfimdb/training_metrics_batch_16_lr_1e-5_drop_0.45.csv > out/cfimdb/batch_16_lr_1e-5_drop_0.45.log 2>&1 &
# echo "started job 3!"

# CUDA_VISIBLE_DEVICES=3 python3 classifier.py --epochs 15 --fine-tune-mode full-model --batch_size 16 --lr 5e-6 --hidden_dropout_prob 0.4 --use_gpu --log_file out/cfimdb/training_metrics_batch_16_lr_5e-6_drop_0.4.csv > out/cfimdb/batch_16_lr_5e-6_drop_0.4.log 2>&1 &
# echo "started job 4!"

# wait for all subprocesses to finish
wait
echo "All jobs finished!"