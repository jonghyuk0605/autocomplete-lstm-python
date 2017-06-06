#!/bin/bash
set -x

if [ $1 -eq 0 ]
then
    python train.py --log_dir log/0 --save_dir save/0/model --text_modeling chr
elif [ $1 -eq 1 ]
then
    python train.py --log_dir log/1 --save_dir save/1/model --text_modeling syl
fi
