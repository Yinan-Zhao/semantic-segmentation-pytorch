#!/bin/bash
python -u train_memory.py --gpus 0,1,2,3 --cfg config/ade20k-memory-resnet18dilated-c1-normKey-batch1.yaml \
2>&1 | tee ./output/output_memory_c1_normKey_batch1.log
