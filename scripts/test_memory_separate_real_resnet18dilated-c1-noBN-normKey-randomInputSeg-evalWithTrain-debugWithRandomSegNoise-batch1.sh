#!/bin/bash
python eval_memory_separate_multipro.py --gpus 0,1,2,3 --debug_with_randomSegNoise --eval_with_train --cfg config/ade20k-memory-separate-real-resnet18dilated-c1-noBN-normKey-randomInputSeg-batch1.yaml \
2>&1 | tee ./output/output_test_memory_separate_real_c1_noBN_normKey_randomInputSeg_evalWithTrain_debugWithRandomSegNoise_batch1.log
