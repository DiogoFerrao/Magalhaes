#!/bin/bash

data=$1
exp_name="distil_$(date +%s)"

cd ..

python distillate.py \
    --project /media/cache/magalhaes/vision/distillations \
    --teacher_weights /media/magalhaes/vision/pretrained/yolov7x_training.pt \
    --teacher_cfg cfg/training/yolov7x.yaml \
    --data data/coco.yaml \
    --batch-size 64 \
    --epochs 150 \
    --workers 18 \
    --kd-ratio 0.5 \
    --hyp data/hyp.distillation.tiny.yaml \
    --device 0 \
    --exp_name $exp_name
