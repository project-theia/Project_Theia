#!/bin/bash

python take_photo.py
sleep 3
python train_and_recog.py
sleep 3
rm result.jpg
