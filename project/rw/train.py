import sys
sys.path.append("..")
import logging, os, sys, json, torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.nn import MSELoss
import pytorch_lightning as pl
from transformers import AutoTokenizer, T5ForConditionalGeneration, AutoConfig, Trainer, TrainingArguments
from pytorch_lightning.callbacks import EarlyStopping
from scipy.stats.stats import pearsonr
from scipy.stats import spearmanr

from pytorch_lightning.callbacks import ModelCheckpoint

import os
from tqdm import tqdm
from model import T5RewardModel
from argparse import ArgumentParser
from utils.params import MODEL_PATH

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def create_comparison_dataset(split="train"):
    if "train" in split:
        dataset = json.load(open("../../data/rw.json" , "r"))["train"]
    else:
        dataset = json.load(open("../../data/rw.json" , "r"))["val"]
    pairs = []
    for sample in tqdm(dataset):
        pair = {}
        prompt = "Summary:"
        chosen_summary = sample["accepted"]
        rejected_summary = sample["rejected"]
        if chosen_summary == rejected_summary:
            continue
        if len(chosen_summary.split()) < 5 or len(rejected_summary.split()) < 5:
            continue
        pair["chosen"] = prompt + "\n" + chosen_summary
        pair["rejected"] = prompt + "\n" + rejected_summary
        pairs.append(pair)
    return pairs

class PairwiseDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_length):
        self.chosen_input_ids = []
        self.chosen_attn_masks = []
        self.rejected_input_ids = []
        self.rejected_attn_masks = []
        for pair in tqdm(pairs):
            chosen, rejected = pair["chosen"], pair["rejected"]
            #print("CHOSEN", "<pad> " + chosen+ "</s>", "REJECTED", "<pad> " + rejected)
            chosen_encodings_dict = tokenizer(
                chosen+ "</s>",
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            rejected_encodings_dict = tokenizer(
                rejected,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            if not torch.all(torch.eq(chosen_encodings_dict["input_ids"], rejected_encodings_dict["input_ids"])).item():
                self.chosen_input_ids.append(chosen_encodings_dict["input_ids"])
                self.chosen_attn_masks.append(chosen_encodings_dict["attention_mask"])
                self.rejected_input_ids.append(rejected_encodings_dict["input_ids"])
                self.rejected_attn_masks.append(rejected_encodings_dict["attention_mask"])

    def __len__(self):
        return len(self.chosen_input_ids)

    def __getitem__(self, idx):
        return (
            self.chosen_input_ids[idx],
            self.chosen_attn_masks[idx],
            self.rejected_input_ids[idx],
            self.rejected_attn_masks[idx],
        )

class DataCollatorReward:
    def __call__(self, data):
        batch = {}
        batch["input_ids"] = torch.cat([f[0] for f in data] + [f[2] for f in data])
        batch["attention_mask"] = torch.cat([f[1] for f in data] + [f[3] for f in data])
        return batch

def compute_metrics(eval_preds):
    chosen_end_scores = eval_preds.predictions[0]  # chosen scores
    rejected_end_scores = eval_preds.predictions[1]  # rejected scores

    result = {}
    acc = sum(chosen_end_scores > rejected_end_scores) / len(rejected_end_scores)
    result["accuracy"] = acc

    return result


class RewardModel (pl.LightningModule):
    def __init__(self, model_path="../sft/model/model.ckp", tokenizer_name=MODEL_PATH, lr=2e-05, model_max_length=512, inference=True):
        super().__init__()
        print("Loading AutoModel [{}]...".format(model_path))
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.model = T5RewardModel(model_path, self.tokenizer, inference=inference)

        self.lr = lr
        self.model_max_length = model_max_length
        
        self.train_y_hat = []
        self.train_y = []
        self.train_loss = []
        self.valid_y_hat = []
        self.valid_y = []
        self.valid_loss = []

        # add pad token
        self.validate_pad_token()
    
    def validate_pad_token(self):
        if self.tokenizer.pad_token is not None:
            return
        if self.tokenizer.sep_token is not None:
            print(f"\tNo PAD token detected, automatically assigning the SEP token as PAD.")
            self.tokenizer.pad_token = self.tokenizer.sep_token
            return
        if self.tokenizer.eos_token is not None:
            print(f"\tNo PAD token detected, automatically assigning the EOS token as PAD.")
            self.tokenizer.pad_token = self.tokenizer.eos_token
            return
        if self.tokenizer.bos_token is not None:
            print(f"\tNo PAD token detected, automatically assigning the BOS token as PAD.")
            self.tokenizer.pad_token = self.tokenizer.bos_token
            return
        if self.tokenizer.cls_token is not None:
            print(f"\tNo PAD token detected, automatically assigning the CLS token as PAD.")
            self.tokenizer.pad_token = self.tokenizer.cls_token
            return
        raise Exception("Could not detect SEP/EOS/BOS/CLS tokens, and thus could not assign a PAD token which is required.")
        
         
        
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids.to(self.device),
                       attention_mask=attention_mask.to(self.device),
                       )

        return outputs

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask = batch["input_ids"], batch["attention_mask"]
        outputs = self(input_ids, attention_mask) 

        loss =  outputs["loss"]
        self.train_loss.append(loss.detach().cpu().numpy())
        self.log('train_loss',loss)

    def on_train_epoch_end(self):
        mean_train_loss = sum(self.train_loss)/len(self.train_loss)
        self.log("train/avg_loss", mean_train_loss, prog_bar=True)

        self.train_loss = []

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask = batch["input_ids"], batch["attention_mask"]
        
        outputs = self(input_ids, attention_mask)
        loss =  outputs["loss"]
        self.valid_loss.append(loss.detach().cpu().numpy())
        self.log('val_loss',loss)


    def on_validation_epoch_end(self):
        mean_valid_loss = sum(self.valid_loss)/len(self.valid_loss)
        self.log("val/avg_loss", mean_valid_loss, prog_bar=True)

        self.valid_loss = []


    def configure_optimizers(self):
        return torch.optim.AdamW([p for p in self.parameters() if p.requires_grad], lr=self.lr, eps=1e-08)



def cli_main():
    pl.seed_everything(1234)

    # args
    parser = ArgumentParser()
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--accumulate_grad_batches', type=int, default=16)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    # data
    max_length = 510
    train_pairs = create_comparison_dataset("train")
    rw_train = PairwiseDataset(train_pairs, tokenizer, max_length=max_length)
    valid_pairs = create_comparison_dataset("valid")
    rw_valid = PairwiseDataset(valid_pairs, tokenizer, max_length=max_length)
    data_collator = DataCollatorReward()


    train_loader = DataLoader(rw_train, batch_size=args.batch_size, num_workers=8, shuffle=True, collate_fn=data_collator, pin_memory=True)
    valid_loader = DataLoader(rw_valid, batch_size=args.batch_size, num_workers=8, shuffle=False, collate_fn=data_collator, pin_memory=True)
    
    model = RewardModel()

    early_stop = EarlyStopping(
            monitor='train_loss',
            patience=4,
            verbose=True,
            mode='max'
        )
        
    trainer = pl.Trainer(
            devices=args.gpus,
            max_epochs=10,
            #max_steps=10,
            callbacks=[early_stop],
            #limit_train_batches=5,
            #limit_val_batches=2,
            accumulate_grad_batches=args.accumulate_grad_batches,
            gradient_clip_val=1.0,
            enable_checkpointing=False
        )

    trainer.fit(model, train_loader, valid_loader)
    trainer.save_checkpoint("./model/model.ckp")


if __name__ == "__main__":
    cli_main()
