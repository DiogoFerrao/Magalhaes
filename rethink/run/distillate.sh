#!/bin/bash

if [ $# -lt 1 ]
then
  echo "Use: $0 NAME"
  exit 0
fi

name=$1
exp_name="${name}_$(date +%s)"

cd ..

python distillate.py \
  --config_path ./config/schreder_distil_yolov7_tiny.json \
  --exp_name $exp_name
