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

from sft_dataset import SFTDataset, my_collate
from pytorch_lightning.callbacks import ModelCheckpoint

import os
from argparse import ArgumentParser
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class TransformerModel (pl.LightningModule):
    def __init__(self, model_name="google/flan-t5-base", lr=2e-05, model_max_length=512):
        super().__init__()
        print("Loading AutoModel [{}]...".format(model_name))
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)

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
        
         
        
    def forward(self, input_ids, attention_mask, labels, decoder_attention_mask):
        outputs = self.model(input_ids=input_ids.to(self.device),
                       attention_mask=attention_mask.to(self.device),
                       labels=labels.to(self.device),
                       decoder_attention_mask=decoder_attention_mask.to(self.device),
                       )

        return outputs

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, labels, decoder_attention_mask = batch
        
        outputs = self(input_ids, attention_mask, labels, decoder_attention_mask)

        loss =  outputs[0]
        self.train_loss.append(loss.detach().cpu().numpy())
        self.log('train_loss',loss)

        return {"loss": loss}

    def on_train_epoch_end(self):
        mean_train_loss = sum(self.train_loss)/len(self.train_loss)
        self.log("train/avg_loss", mean_train_loss, prog_bar=True)

        self.train_loss = []

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, labels, decoder_attention_mask = batch
        
        outputs = self(input_ids, attention_mask, labels, decoder_attention_mask)

        loss =  outputs[0]
        self.valid_loss.append(loss.detach().cpu().numpy())
        self.log('valid_loss',loss)

        return {"loss": loss}


    def on_validation_epoch_end(self):
        mean_valid_loss = sum(self.valid_loss)/len(self.valid_loss)
        self.log("valid/avg_loss", mean_valid_loss, prog_bar=True)

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

    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    # data
    sft_train = SFTDataset(path="../../data/sft.json", tokenizer=tokenizer, split="train")
    sft_val = SFTDataset(path="../../data/sft.json", tokenizer=tokenizer, split="val")

    train_loader = DataLoader(sft_train, batch_size=args.batch_size, num_workers=8, shuffle=True, collate_fn=my_collate, pin_memory=True)
    val_loader = DataLoader(sft_train, batch_size=args.batch_size, num_workers=8, shuffle=False, collate_fn=my_collate, pin_memory=True)


    model = TransformerModel()

    early_stop = EarlyStopping(
            monitor='valid/loss',
            patience=3,
            verbose=True,
            mode='max'
        )
        
    trainer = pl.Trainer(
            devices=args.gpus,
            max_epochs=10,
            callbacks=[early_stop],
            #limit_train_batches=5,
            #limit_val_batches=2,
            accumulate_grad_batches=args.accumulate_grad_batches,
            gradient_clip_val=1.0,
            enable_checkpointing=False
        )

    trainer.fit(model, train_loader, val_loader)
    trainer.save_checkpoint("./model/")

if __name__ == "__main__":
    cli_main()
