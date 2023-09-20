#!/bin/bash

if [ $# -lt 1 ]
then
  echo "Use: $0 CONFIG NAME"
  exit 0
fi

config=$1
name=$2
exp_name="${name}_$(date +%s)"

cd ..

python train.py \
  --config_path $config \
  --exp_name $exp_name
