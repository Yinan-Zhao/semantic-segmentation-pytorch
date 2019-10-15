#!/bin/bash
srun --mpi=pmi2 -p $1 -n1 --gres=gpu:8 --ntasks-per-node=1 --cpus-per-task=5 \
python -u train_memory_separate.py --gpus 0,1,2,3 --cfg config/ade20k-memory-separate-real-resnet18dilated-c1-randomMemoryBias-scale1e1-batch1.yaml \
2>&1 | tee ./output/output_memory_separate_real_c1_randomMemoryBias_scale1e1_batch1.log
