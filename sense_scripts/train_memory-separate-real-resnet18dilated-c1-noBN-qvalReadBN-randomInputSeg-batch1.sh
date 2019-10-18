#!/bin/bash
srun --mpi=pmi2 -p $1 -n1 --gres=gpu:8 --ntasks-per-node=1 --cpus-per-task=5 \
python -u train_memory_separate.py --gpus 0,1,2,3 --cfg config/ade20k-memory-separate-real-resnet18dilated-c1-noBN-qvalReadBN-randomInputSeg-batch1.yaml \
2>&1 | tee ./output/output_memory_separate_real_c1_noBN_qvalReadBN_randomInputSeg_batch1.log
