#!/bin/bash
nvidia-smi
srun --mpi=pmi2 -p $1 -n1 --gres=gpu:8 --ntasks-per-node=1 --cpus-per-task=5 \
python eval_memory_separate_multipro.py --gpus 0,1,2,3,4,5,6,7 --debug_with_double_random --cfg config/ade20k-memory-separate-real-resnet18dilated-c1-noAlign-batch1.yaml \
2>&1 | tee ./output/output_test_memory_separate_real_c1_noAlign_debugWithDoubleRandom_batch1.log
