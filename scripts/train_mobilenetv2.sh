#!/bin/bash
export PATH=/scratch/cluster/alexzhao/anaconda3/bin:$PATH
cd /scratch/cluster/alexzhao/semantic-segmentation-pytorch
source activate seg_pytorch
python train.py --gpus 0 --cfg config/ade20k-mobilenetv2dilated-c1_deepsup.yaml
