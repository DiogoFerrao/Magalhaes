#!/bin/bash

if [ $# -lt 1 ]
then
  echo "Use: $0 DATA"
  exit 0
fi

data=$1
exp_name="finetune_$(date +%s)"
savedir="/media/cache/magalhaes/vision/checkpoints"

cd ..

python train.py \
    --weights /media/magalhaes/vision/pretrained/yolov4-csp.weights \
    --cfg ./models/yolov4-csp-schreder.cfg \
    --data $data \
    --project $savedir \
    --batch-size 12 \
    --workers 14 \
    --device 2 \
    --name $exp_name
