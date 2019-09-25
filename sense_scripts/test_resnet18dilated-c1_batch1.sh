#!/bin/bash
nvidia-smi
srun --mpi=pmi2 -p $1 -n1 --gres=gpu:8 --ntasks-per-node=1 --cpus-per-task=5 \
python eval_multipro.py --gpus 0,1,2,3 --cfg config/ade20k-resnet18dilated-c1_batch1.yaml \
2>&1 | tee ./output_test_c1_batch1.log
