#!/bin/bash
python eval_memory_separate_multipro.py --gpus 0,1,2,3 --cfg config/ade20k-memory-separate-real-resnet18dilated-c1-noBN-normKey-batch1.yaml \
2>&1 | tee ./output/output_test_memory_separate_real_c1_noBN_normKey_batch1_danna.log
