import os
import sys
sys.path.append("../")
sys.path.append("../sft")
sys.path.append("../rw")
from typing import List

import torch
import json
from datasets import load_dataset
from rw.train import RewardModel
from sft.train import TransformerModel
from tqdm import tqdm
from transformers import AutoTokenizer
from utils.params import MODEL_PATH

import trlx
from trlx.data.default_configs import (
    ILQLConfig,
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainConfig,
    TRLConfig,
)

REWARD_CHECKPOINT_PATH = "../rw/model/model.ckp"
if not os.path.exists(REWARD_CHECKPOINT_PATH):
    print("There is no reward model!")

SFT_MODEL_PATH = "../sft/model/model.ckp"

HF_SFT_MODEL_PATH="../sft/hf_model/"

print("Converting PL model to HF model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.save_pretrained(HF_SFT_MODEL_PATH)
sft_model = TransformerModel.load_from_checkpoint(SFT_MODEL_PATH)
sft_model.model.save_pretrained(HF_SFT_MODEL_PATH)
del sft_model
del tokenizer

default_config = TRLConfig(
    train=TrainConfig(
        seq_length=510,
        epochs=100,
        total_steps=10000,
        batch_size=1,
        checkpoint_interval=5000,
        eval_interval=5000,
        pipeline="PromptPipeline",
        trainer="AccelerateILQLTrainer",
        checkpoint_dir="ilql_summarize_t5"
    ),
    model=ModelConfig(
        model_path=HF_SFT_MODEL_PATH,
        num_layers_unfrozen=-1,
        model_arch_type="seq2seq"
    ),
    tokenizer=TokenizerConfig(
        tokenizer_path=HF_SFT_MODEL_PATH,
        truncation_side="left"
    ),
    optimizer=OptimizerConfig(
        name="adamw",
        kwargs={
            "lr": 1.0e-6,
            "betas": [0.9, 0.999],
            "eps": 1.0e-8,
            "weight_decay": 1.0e-6,
        },
    ),
    scheduler=SchedulerConfig(
        name="cosine_annealing",
        kwargs={
            "T_max": 5000,
            "eta_min": 1.0e-6,
        },
    ),
    method=ILQLConfig(
        name="ilqlconfig",
        tau=0.6,
        gamma=0.99,
        cql_scale=0.1,
        awac_scale=1,
        alpha=0.0001,
        beta=0,
        steps_for_target_q_sync=1,
        two_qs=True,
        gen_kwargs=dict(max_new_tokens=256, top_k=50, beta=[1, 2, 3], temperature=1.0)
    ),
)


if __name__ == "__main__":
    hparams = {}
    config = TRLConfig.update(default_config, hparams)

    # Load the pre-trained reward model
    rw_tokenizer = AutoTokenizer.from_pretrained(HF_SFT_MODEL_PATH)
    rw_tokenizer.pad_token = rw_tokenizer.eos_token
    pl_model = RewardModel.load_from_checkpoint(REWARD_CHECKPOINT_PATH)
    # set model for inference
    rw_model = pl_model.model
    rw_model.half()
    rw_model.eval()
    rw_device = "cuda"
    rw_model.to(rw_device)

    print("Starting training...")

    def reward_fn(samples):
        scores_list = []
        batch_size = 1
        for i in range(0, len(samples), batch_size):
            sub_samples = samples[i : i + batch_size]
            sub_samples = [ chosen + "</s>" for chosen in sub_samples]
            encodings_dict = rw_tokenizer(sub_samples,
                                          truncation=True,
                                          max_length=config.train.seq_length,
                                          padding="max_length",
                                          return_tensors="pt")
            input_ids = encodings_dict["input_ids"].to(rw_device)
            attn_masks = encodings_dict["attention_mask"].to(rw_device)
            input_ids = input_ids.repeat(2, 1)
            attn_masks = attn_masks.repeat(2, 1)
            with torch.no_grad():
                sub_scores = rw_model(input_ids=input_ids,
                                      attention_mask=attn_masks)
            scores_list.append(sub_scores["chosen_end_scores"])
        scores = torch.cat(scores_list, dim=0)

        return scores


    def preprocess(sample):
        sample["prompt_output"] = [
                [sample["prompt"] + "Summary:", sample["chosen"]],
                [sample["prompt"] + "Summary:", sample["rejected"]]
        ]
        sample["reward"] = [1, -1]
        return sample

    train_dataset = json.load(open("./saved_summaries_train.json", "r"))
    valid_dataset = json.load(open("./saved_summaries_valid.json", "r"))

    # convert json to dataset huggingface
    train_dataset = load_dataset("json", data_files="./saved_summaries_train.json", field="train")
    valid_dataset = load_dataset("json", data_files="./saved_summaries_valid.json", field="valid")

    dataset_train = train_dataset.map(preprocess)
    dataset_valid = valid_dataset.map(preprocess)

    train_prompts_outputs = sum(dataset_train["train"]["prompt_output"], [])
    train_rewards = sum(dataset_train["train"]["reward"], [])
    eval_prompts = list(dataset_valid["train"]["prompt"])

    ind = 0
    for sent in train_prompts_outputs:
        if len(sent[1]) == 0:
             del train_prompts_outputs[ind]
             del train_rewards[ind]
        ind += 1



    trainer = trlx.train(
        dataset=(train_prompts_outputs, train_rewards),
        metric_fn=lambda samples, **kwargs: {"rewards": reward_fn(samples)},
        eval_prompts=eval_prompts[0:1000],  # sampling 1000 validation prompts for evaluation speed in training
        config=config,
    )

    trainer.save_pretrained('model')
