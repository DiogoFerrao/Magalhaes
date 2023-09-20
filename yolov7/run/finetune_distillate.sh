#!/bin/bash

data=$1
exp_name="distil_schreder_$(date +%s)"

cd ..

python distillate.py \
    --project /media/cache/magalhaes/vision/distillations \
    --teacher_weights /media/magalhaes/vision/checkpoints/yolov7_1681719082/weights/best.pt \
    --teacher_cfg cfg/training/yolov7.yaml \
    --data data/schreder.yaml \
    --batch-size 64 \
    --epochs 300 \
    --workers 14 \
    --kd-ratio 0.5 \
    --hyp data/hyp.distillation.tiny.yaml \
    --device 2 \
    --exp_name $exp_name
