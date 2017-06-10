#!/bin/bash
set -x

if [ $1 -eq 0 ]
then
    python train.py --log_dir log/0 --save_dir save/model_0.ckpt --text_modeling chr #--load_dir save/model_0.ckpt
elif [ $1 -eq 1 ]
then
    python train.py --log_dir log/1 --save_dir save/model_1.ckpt --text_modeling syl #--load_dir save/model_1.ckpt
fi
