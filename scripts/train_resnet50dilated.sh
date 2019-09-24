#!/bin/bash
export PATH=/scratch/cluster/alexzhao/anaconda3/bin:$PATH
cd /scratch/cluster/alexzhao/semantic-segmentation-pytorch
source activate seg_pytorch
python train.py --gpus 0-3 --cfg config/ade20k-resnet50dilated-ppm_deepsup.yaml
