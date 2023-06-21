import sys
sys.path.append("..")
sys.path.append("../sft")
import torch
from torch import nn
from tqdm import tqdm
import pytorch_lightning as pl
from transformers import T5ForConditionalGeneration, T5Tokenizer
from sft.train import TransformerModel 

class T5RewardModel(pl.LightningModule):
    def __init__(self, model_path, tokenizer, inference=True):
        super().__init__()
        model_pl = TransformerModel.load_from_checkpoint(model_path) 
        model = model_pl.model
        self.config = model.config
        self.config.n_embd = self.config.hidden_size if hasattr(self.config, "hidden_size") else self.config.n_embd
        self.encoder = model.encoder
        self.v_head = nn.Linear(self.config.n_embd, 1, bias=False)
        self.PAD_ID = tokenizer(tokenizer.pad_token)["input_ids"][0]
        self.inference = inference

    def forward(self,
                input_ids=None,
                attention_mask=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                return_dict=False,
                output_attentions=False,
                output_hidden_states=False):
        
        loss = None
        transformer_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict)
        hidden_states = transformer_outputs[0]
        
        rewards = self.v_head(hidden_states).squeeze(-1)
        chosen_end_scores = []
        rejected_end_scores = []

        assert len(input_ids.shape) == 2
        bs = input_ids.shape[0] // 2
        chosen = input_ids[:bs]
        rejected = input_ids[bs:]
        chosen_rewards = rewards[:bs]
        rejected_rewards = rewards[bs:]

        loss = 0
        for i in range(bs):
            if torch.all(torch.eq(chosen[i] ,rejected[i])).item():
                c_inds = (chosen[i] == self.PAD_ID).nonzero()
                c_ind = c_inds[0].item() if len(c_inds) > 0 else chosen.shape[1]
                chosen_end_scores.append(chosen_rewards[i, c_ind-1])
                continue
        
            # check if there is any padding otherwise take the length of the sequence
            c_inds = (chosen[i] == self.PAD_ID).nonzero()
            c_ind = c_inds[0].item() if len(c_inds) > 0 else chosen.shape[1]
            r_inds = (rejected[i] == self.PAD_ID).nonzero()
            r_ind = r_inds[0].item() if len(r_inds) > 0 else rejected.shape[1]
            end_ind = max(c_ind, r_ind)

            # retrive the first index where trajectories diverge
            divergence_ind = (chosen[i] != rejected[i]).nonzero()[0]
            assert divergence_ind > 0

            #index the correct rewards
            c_truncated_reward = chosen_rewards[i][divergence_ind:end_ind]
            r_truncated_reward = rejected_rewards[i][divergence_ind:end_ind]

            #append the last rewards to  the list of end scores
            chosen_end_scores.append(c_truncated_reward[-1])
            rejected_end_scores.append(r_truncated_reward[-1])

            #compute loss based on truncated rewards
            loss += -torch.log(torch.sigmoid(c_truncated_reward - r_truncated_reward)).mean()
        loss = loss / bs

        if not self.inference:
            rejected_end_scores = torch.stack(rejected_end_scores)

        if self.inference:
            chosen_end_scores = torch.stack(chosen_end_scores)
            return {"chosen_end_scores": chosen_end_scores}

        return {
            "loss": loss,
            "chosen_end_scores": chosen_end_scores,
            "rejected_end_scores": rejected_end_scores
            }
