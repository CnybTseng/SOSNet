#!/bin/sh

python train.py \
    --batch-size 1024 \
    --learning-rate 0.5 \
    --train-dir /data/tseng/dataset/ILSVRC2012/data/train \
    --val-dir /data/tseng/dataset/ILSVRC2012/data/val