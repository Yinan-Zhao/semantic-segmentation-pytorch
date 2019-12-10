#!/bin/bash
python eval_memory_separate_multipro.py --gpus 0,1,2,3 --cfg $1 --ref_val_start 0 --ref_val_end 1 --eval_att_voting \
2>&1 | tee $2_vote.log
python eval_memory_separate_multipro.py --gpus 0,1,2,3 --cfg $1 --ref_val_start 1 --ref_val_end 2 --eval_att_voting \
2>&1 | tee $2_vote_ref1.log
python eval_memory_separate_multipro.py --gpus 0,1,2,3 --cfg $1 --ref_val_start 10 --ref_val_end 11 --eval_att_voting \
2>&1 | tee $2_vote_ref10.log
python eval_memory_separate_multipro.py --gpus 0,1,2,3 --cfg $1 --ref_val_start 100 --ref_val_end 101 --eval_att_voting \
2>&1 | tee $2_vote_ref100.log
python eval_memory_separate_multipro.py --gpus 0,1,2,3 --cfg $1 --ref_val_start 0 --ref_val_end 1 --debug_with_gt \
--eval_att_voting 2>&1 | tee $2_vote_gt.log
python eval_memory_separate_multipro.py --gpus 0,1,2,3 --cfg $1 --ref_val_start 0 --ref_val_end 1 \
--debug_with_double_complete_random --eval_att_voting 2>&1 | tee $2_vote_double_complete_random.log


