#!/bin/bash
python -u debug_train_memory_separate.py --gpus 0 --cfg config/ade20k-memory-separate-real-resnet18dilated-c1-noBN-normKey-batch1-debug.yaml \
2>&1 | tee ./output/output_memory_separate_real_c1_noBN_normKey_batch1_debug.log
