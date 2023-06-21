import os
import json
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split


f = open("../data/data.json", "r")

lines = json.load(f, strict=False)

filtered_data = []
count = 0
for line in lines["data"]:
    if line["NlgFeedback"] == 0:
        if len(line["NlgSummary"]) > 0:
            filtered_data.append({"prompt" : line["Body"], "chosen" : line["NlgSummary"]})
            count += 1


N = len(filtered_data)
data_sft  = []
data_rw = []
data_rl = []


print("Split SFT...")
# Dataset for Supervised Finetunning
for el in tqdm(filtered_data[:int(N/3)]):
    data_sft.append({"prompt" : el["prompt"], "chosen" : el["chosen"]})
data_sft = np.array(data_sft)
data_train_sft, data_val_sft  = train_test_split(data_sft, test_size=0.1)
d = {"train": data_train_sft, "val": data_val_sft}
j = json.dumps({k: v.tolist() for k, v in d.items()}, indent=4)
with open("../data/sft.json", "w") as outfile:
    outfile.write(j)

print("Split RW...")
# Dataset for Supervised Finetunning
for el in tqdm(filtered_data[int(N/3)+1:int(2*N/3)]):
    data_rw.append({"prompt" : el["prompt"], "chosen" : el["chosen"]})
data_rw = np.array(data_rw)
data_train_rw, data_val_rw  = train_test_split(data_rw, test_size=0.1)
d = {"train": data_train_rw, "val": data_val_rw}
j = json.dumps({k: v.tolist() for k, v in d.items()}, indent=4)
with open("../data/rw.json", "w") as outfile:
    outfile.write(j)

print("Split RL...")
# Dataset for Supervised Finetunning
for el in tqdm(filtered_data[int(2*N/3)+1:N]):
    data_rl.append({"prompt" : el["prompt"], "chosen" : el["chosen"]})
data_rl = np.array(data_rl)
data_train_rl, data_val_rl  = train_test_split(data_rl, test_size=0.1)
d = {"train": data_train_rl, "val": data_val_rl}
j = json.dumps({k: v.tolist() for k, v in d.items()}, indent=4)
with open("../data/rl.json", "w") as outfile:
    outfile.write(j)

