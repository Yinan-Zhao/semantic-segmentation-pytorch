#!/bin/bash
python -u train_memory_separate.py --gpus 0,1,2,3 --cfg config/ade20k-memory-separate-real-hrnetv2-c1-normKey-pick01in10.yaml \
2>&1 | tee ./output/output_memory_separate_real_hrnetv2_c1_normKey_pick01in10.log

