#!/bin/bash
python -u train_memory_separate.py --gpus 0,1,2,3 --cfg config/ade20k-memory-separate-real-resnet18dilated-c1-normKey-featAggre-noLabel-batch1.yaml \
2>&1 | tee ./output/output_memory_separate_real_c1_normKey_featAggre_noLabel_batch1.log
