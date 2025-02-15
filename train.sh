#!/bin/bash
cd /home/xin/aws/Bi-FRN-main/experiments/cars/BiFRN/ResNet-18
python train.py\
    --opt sgd \
    --lr 1e-2 \
    --gamma 1e-1 \
    --epoch 400 \
    --stage 3 \
    --val_epoch 20 \
    --weight_decay 5e-4 \
    --nesterov \
    --train_way 15 \
    --train_shot 5 \
    --train_transform_type 0 \
    --test_transform_type 0 \
    --test_shot 1 5 \
    --backbone resnet18 \
    --gpu 0