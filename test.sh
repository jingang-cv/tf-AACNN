#!bin/bash
mkdir result_attr_test/
mkdir result_attr_test/gt/
CUDA_VISIBLE_DEVICES=$1 python main.py --experiment_name celebA_demo
