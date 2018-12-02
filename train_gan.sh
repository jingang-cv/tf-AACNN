#!bin/bash
CUDA_VISIBLE_DEVICES=$1 python main.py --experiment_name celebA_demo --train True --with_gan True
