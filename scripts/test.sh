#!/bin/bash
python eval_memory_separate_multipro.py --gpus 0,1,2,3 --cfg $1 --ref_val_start 0 --ref_val_end 1 \
2>&1 | tee $2.log
python eval_memory_separate_multipro.py --gpus 0,1,2,3 --cfg $1 --ref_val_start 0 --ref_val_end 2 \
2>&1 | tee $2_ref0-2.log
python eval_memory_separate_multipro.py --gpus 0,1,2,3 --cfg $1 --ref_val_start 0 --ref_val_end 3 \
2>&1 | tee $2_ref0-3.log
python eval_memory_separate_multipro.py --gpus 0,1,2,3 --cfg $1 --ref_val_start 1 --ref_val_end 2 \
2>&1 | tee $2_ref1.log
python eval_memory_separate_multipro.py --gpus 0,1,2,3 --cfg $1 --ref_val_start 10 --ref_val_end 11 \
2>&1 | tee $2_ref10.log
python eval_memory_separate_multipro.py --gpus 0,1,2,3 --cfg $1 --ref_val_start 100 --ref_val_end 101 \
2>&1 | tee $2_ref100.log
python eval_memory_separate_multipro.py --gpus 0,1,2,3 --cfg $1 --ref_val_start 200 --ref_val_end 201 \
2>&1 | tee $2_ref200.log
python eval_memory_separate_multipro.py --gpus 0,1,2,3 --cfg $1 --ref_val_start 0 --ref_val_end 1 --debug_with_gt \
2>&1 | tee $2_gt.log
python eval_memory_separate_multipro.py --gpus 0,1,2,3 --cfg $1 --ref_val_start 0 --ref_val_end 1 \
--debug_with_double_complete_random 2>&1 | tee $2_double_complete_random.log
python eval_memory_separate_multipro.py --gpus 0,1,2,3 --cfg $1 --ref_val_start 0 --ref_val_end 4 \
2>&1 | tee $2_ref0-4.log


