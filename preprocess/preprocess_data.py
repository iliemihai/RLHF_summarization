import os
import json
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split


f = open("../data/json_data.json", "r")

lines = json.load(f)

filtered_data = []
count = 0
for line in lines["data"]:
    if line["NlgFeedback"] == 0:
        #print("NEWS: ", line["Body"])
        #print("SUMMARY: ", line["NlgSummary"])
        filtered_data.append([line["Body"], line["NlgSummary"]])
        count += 1
        #print("===============================================================")


N = len(filtered_data)
X_sft, Y_sft = [], []
X_rw, Y_rw = [], []
X_rl, Y_rl = [], []


print("Split SFT...")
# Dataset for Supervised Finetunning
for el in tqdm(filtered_data[:int(N/3)]):
    X_sft.append(el[0])
    Y_sft.append(el[1])
X_sft = np.array(X_sft)
Y_sft = np.array(Y_sft)
X_train_sft, X_val_sft, Y_train_sft, Y_val_sft  = train_test_split(X_sft, Y_sft, test_size=0.1)
print(X_val_sft)
d = {"x_train": X_train_sft, "y_train": Y_train_sft, "x_val": X_val_sft, "y_val": Y_val_sft}
j = json.dumps({k: v.tolist() for k, v in d.items()})
with open("../data/sft.json", "w") as outfile:
    outfile.write(j)

print("Split RW...")
# Dataset for Reward Model
for el in tqdm(filtered_data[int(N/3)+1:int(2*N/3)]):
    X_rw.append(el[0])
    Y_rw.append(el[1])
X_rw = np.array(X_rw)
Y_rw = np.array(Y_rw)
X_train_rw, X_val_rw, Y_train_rw, Y_val_rw  = train_test_split(X_rw, Y_rw, test_size=0.1)
d = {"x_train": X_train_rw, "y_train": Y_train_rw, "x_val": X_val_rw, "y_val": Y_val_rw}
j = json.dumps({k: v.tolist() for k, v in d.items()})
with open("../data/rw.json", "w") as outfile:
    outfile.write(j)


print("Split RL...")
# Dataset for RL
for el in tqdm(filtered_data[int(2*N/3)+1:N]):
    X_rl.append(el[0])
    Y_rl.append(el[1])
X_rl = np.array(X_rl)
Y_rl = np.array(Y_rl)
X_train_rl, X_val_rl, Y_train_rl, Y_val_rl  = train_test_split(X_rl, Y_rl, test_size=0.1)
d = {"x_train": X_train_rl, "y_train": Y_train_rl, "x_val": X_val_rl, "y_val": Y_val_rl}
j = json.dumps({k: v.tolist() for k, v in d.items()})
with open("../data/rl.json", "w") as outfile:
    outfile.write(j)



