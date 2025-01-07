#!/bin/bash

python train_model2.py \
    --opt sgd \
    --lr 1e-1 \
    --gamma 1e-1 \
    --epoch 1 \
    --stage 2 \
    --val_epoch 1 \
    --weight_decay 5e-4 \
    --nesterov \
    --train_way 5 \
    --train_shot 1 \
    --train_transform_type 0 \
    --test_shot 1 5 \
    --pre \
    --gpu 0\

python train_ASCO.py

