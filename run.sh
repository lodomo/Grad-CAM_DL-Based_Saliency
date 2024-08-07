#!/bin/bash

export TF_ENABLE_ONEDNN_OPTS=0
export TF_CPP_MIN_LOG_LEVEL=4

options=(1 2 3)
images=(gc1.jpg gc2.jpg gc3.jpg gc4.jpg gc5.jpg)

for i in "${images[@]}"
do
    for j in "${options[@]}"
    do
        echo "Running python main.py ./images/$i $j"
        python main.py ./images/$i $j >> ./output/trials.txt
    done
done
