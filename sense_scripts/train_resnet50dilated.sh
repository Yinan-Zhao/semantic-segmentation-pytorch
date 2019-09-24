#!/bin/bash
nvidia-smi
srun --mpi=pmi2 -p $1 -n1 --gres=gpu:8 --ntasks-per-node=1 --cpus-per-task=5 \
python train.py --gpus 0,1,2,3 --cfg config/ade20k-resnet50dilated-ppm_deepsup.yaml \
2>&1 | tee ./output.log
