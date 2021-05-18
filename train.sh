#!/bin/sh

python train.py \
    --batch-size 512 \
    --learning-rate 0.25 \
    --train-dir /data/tseng/dataset/ILSVRC2012/data/train \
    --val-dir /data/tseng/dataset/ILSVRC2012/data/val