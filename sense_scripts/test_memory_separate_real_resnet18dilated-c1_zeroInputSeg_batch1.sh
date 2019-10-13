#!/bin/bash
nvidia-smi
srun --mpi=pmi2 -p $1 -n1 --gres=gpu:8 --ntasks-per-node=1 --cpus-per-task=5 \
python -m pdb eval_memory_separate_multipro.py --gpus 0 --cfg config/ade20k-memory-separate-real-resnet18dilated-c1-zeroInputSeg-batch1.yaml
