#!/bin/bash
python -u train_memory_separate.py --gpus 0,1,2,3 --cfg config/ade20k-memory-separate-real-resnet18dilated-c1-c1Mask-normKey-pick01in10-batch1.yaml \
2>&1 | tee ./output/output_memory_separate_real_c1_c1Mask_normKey_pick01in10_batch1.log

'''python -u eval_memory_separate_multipro.py --gpus 0,1,2,3 --cfg config/ade20k-memory-separate-real-resnet18dilated-c1-c1Mask-normKey-pick01in10-batch1.yaml 2>&1 | tee output_test_memory_separate_real_c1_c1Mask_normKey_pick01in10_batch1.log

python -u eval_memory_separate_multipro.py --gpus 0,1,2,3 --debug_with_gt --cfg config/ade20k-memory-separate-real-resnet18dilated-c1-c1Mask-normKey-pick01in10-batch1.yaml 2>&1 | tee output_test_memory_separate_real_c1_c1Mask_normKey_pick01in10_batch1_gt.log

python -u eval_memory_separate_multipro.py --gpus 0,1,2,3 --debug_with_double_complete_random --cfg config/ade20k-memory-separate-real-resnet18dilated-c1-c1Mask-normKey-pick01in10-batch1.yaml 2>&1 | tee output_test_memory_separate_real_c1_c1Mask_normKey_pick01in10_batch1_double_complete_random.log'''
