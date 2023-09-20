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
    --batch-size 8 \
    --device 0 \
    --save-incorrect \
    --cfg ./models/yolov4-csp.cfg \
    --names ./data/schreder_pretrained.names
