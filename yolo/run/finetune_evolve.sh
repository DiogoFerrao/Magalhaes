#!/bin/bash

if [ $# -lt 1 ]
then
  echo "Use: $0 SAVE_DIR"
  exit 0
fi

exp_name="finetune_$(date +%s)"
savedir=$1

cd ..

python train.py \
    --weights /media/magalhaes/vision/pretrained/yolov4-csp.weights \
    --cfg ./models/yolov4-csp-schreder.cfg \
    --data ./data/schreder.yaml \
    --project $savedir \
    --batch-size 12 \
    --weights /media/magalhaes/yolov4-csp.weights \
    --name $exp_name \
    --evolve
