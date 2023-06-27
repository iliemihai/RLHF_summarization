#!/bin/bash

# Split data
python3.8 preprocess/preprocess_data.py


# Finetune with Supervised learning
cd project/sft/
python3.8 train.py

cd project/sft/
cd ../rw
# create Reward model dataset
python3.8 generate_rejected_summaries.py
# train reward model
python3.8 train.py #--inference

cd ../rl
# create Reinforcement Learning model dataset
python3.8 generate_rl_dataset.py
# finetune with RLHF
python3.8 train.py --inference
