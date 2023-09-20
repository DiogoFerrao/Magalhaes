#!/bin/bash

if [ $# -lt 1 ]
then
  echo "Use: $0 CONFIG"
  exit 0
fi

config=$1

cd ..

python evaluate.py \
  --config_path $config
