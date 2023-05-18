import logging, os, sys, json, torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.nn import MSELoss
import pytorch_lightning as pl
from transformers import AutoTokenizer, AutoModel, AutoConfig, Trainer, TrainingArguments
from pytorch_lightning.callbacks import EarlyStopping
from scipy.stats.stats import pearsonr
from scipy.stats import spearmanr

from sft_dataset import SFTDataset, my_collate
from pytorch_lightning.callbacks import ModelCheckpoint

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class TransformerModel (pl.LightningModule):
    def __init__(self, model_name="google/flan-t5-base", lr=2e-05, model_max_length=512):
        super().__init__()
        print("Loading AutoModel [{}]...".format(model_name))
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

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

    def training_epoch_end(self, outputs):
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


    def validation_epoch_end(self, outputs):
        mean_valid_loss = sum(self.valid_loss)/len(self.valid_loss)
        self.log("valid/avg_loss", mean_valid_loss, prog_bar=True)

        self.valid_loss = []


    def configure_optimizers(self):
        return torch.optim.AdamW([p for p in self.parameters() if p.requires_grad], lr=self.lr, eps=1e-08)




def cli_main():
    pl.seed_everything(1234)

    # args
    parser = ArgumentParser()
    parser.add_argument("--batch_size", default=16, type=int)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = LitClassifier.add_model_specific_args(parser)

    # data
    sft_train = SFTDataset(path="../../data/sft.json", tokenizer=tokenizer, split="train")
    sft_val = SFTDataset(path="../../data/sft.json", tokenizer=tokenizer, split="val")

    train_loader = DataLoader(sft_train, batch_size=args.batch_size, num_workers=1, shuffle=True, collate_fn=my_collate, pin_memory=True)
    val_loader = DataLoader(sft_train, batch_size=args.batch_size, num_workers=1, shuffle=False, collate_fn=my_collate, pin_memory=True)


    model = TransformerModel()

    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    cli_main()
