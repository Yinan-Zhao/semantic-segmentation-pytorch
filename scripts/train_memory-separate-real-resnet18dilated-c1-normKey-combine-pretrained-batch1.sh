#!/bin/bash
python -u train_memory_separate.py --gpus 0,1,2,3 --memory_enc_pretrained --cfg config/ade20k-memory-separate-real-resnet18dilated-c1-normKey-combine-pretrained-batch1.yaml \
2>&1 | tee ./output/output_memory_separate_real_c1_normKey_combine_pretrained_batch1.log
