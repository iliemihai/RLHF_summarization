#!/bin/bash

# Split data
python3.8 preprocess/preprocess_data.py


# Finetune with Supervised learning
cd project/sft/
python3.8 train.py
