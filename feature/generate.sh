#!/bin/bash
srun --mpi=pmi2 -p $1 -n1 --gres=gpu:1 --ntasks-per-node=1 --cpus-per-task=5 \
python -i generate_feature.py
