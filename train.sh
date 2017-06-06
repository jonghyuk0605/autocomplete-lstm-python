#!/bin/bash
set -x

if [ $1 -eq 0 ]
then
    python train.py --log_dir log --save_dir save/model --text_modeling chr
elif [ $1 -eq 1 ]
then
    python train.py --log_dir log --save_dir save/model --text_modeling syl
fi
