#!/bin/bash

if [ $# -lt 1 ]
then
  echo "Use: $0 DATA"
  exit 0
fi

data=$1
exp_name="yolov7_tiny_$(date +%s)"

cd ..

python train.py \
  --workers 8 \
  --device 3 \
  --batch-size 64 \
  --data $data \
  --cfg ./cfg/training/yolov7-tiny-schreder.yaml \
  --weights /media/magalhaes/vision/pretrained/yolov7-tiny.pt \
  --name $exp_name \
  --hyp data/hyp.schreder.finetuning.tiny.yaml
