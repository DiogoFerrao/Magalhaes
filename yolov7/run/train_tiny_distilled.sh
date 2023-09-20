#!/bin/bash

if [ $# -lt 1 ]
then
  echo "Use: $0 DATA WEIGHTS"
  exit 0
fi

data=$1
weights=$2
exp_name="yolov7_tiny_$(date +%s)"

cd ..

python train.py \
  --workers 12 \
  --device 0 \
  --batch-size 64 \
  --data $data \
  --cfg ./cfg/training/yolov7-tiny-schreder.yaml \
  --weights $weights \
  --name $exp_name \
  --distilled \
  --hyp data/hyp.schreder.finetuning.tiny.yaml
