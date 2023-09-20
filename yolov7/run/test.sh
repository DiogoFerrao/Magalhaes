#!/bin/bash

if [ $# -lt 2 ]
then
  echo "Use: $0 WEIGHTS DATA"
  exit 0
fi

weights=$1
data=$2

cd ..

python test.py \
    --weights $weights \
    --data $data \
    --conf-thres 0.05 \
    --iou-thres 0.55 \
    --batch-size 32 \
    --save-stats \
    --device 0
