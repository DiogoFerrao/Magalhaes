#!/bin/bash

if [ $# -lt 1 ]
then
  echo "Use: $0 DATA"
  exit 0
fi

data=$1
exp_name="yolov7_$(date +%s)"

cd ..

python train.py \
  --workers 14 \
  --device 2 \
  --batch-size 32 \
  --data $data \
  --cfg ./cfg/training/yolov7-schreder.yaml \
  --weights /media/magalhaes/vision/pretrained/yolov7_training.pt \
  --name $exp_name \
  --hyp data/hyp.schreder.finetuning.yaml
