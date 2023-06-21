import sys
sys.path.append("..")
import json
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration
from utils.params import MODEL_PATH

d = json.load(open("../../data/rl.json"))

model_name = "../sft/hf_model/"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name, device_map="auto", load_in_8bit=True)


print("Generate for train...")
for line in tqdm(d["train"][:]):
    prompt = "{0} Summary:".format(line["prompt"])
    input_ids = tokenizer(prompt, padding="max_length", max_length=512, truncation=True, return_tensors="pt").input_ids.to("cuda")
    outputs = model.generate(input_ids, penalty_alpha=0.6, top_k=4, max_length=512)
    generated_summary = tokenizer.decode(outputs[0])
    line["rejected"] = generated_summary

print("Generate for val...")
for line in tqdm(d["val"][:]):
    prompt = "{0} Summary:".format(line["prompt"])
    input_ids = tokenizer(prompt, padding="max_length", max_length=512, truncation=True, return_tensors="pt").input_ids.to("cuda")
    outputs = model.generate(input_ids, penalty_alpha=0.6, top_k=4, max_length=512)
    generated_summary = tokenizer.decode(outputs[0])
    line["rejected"] = generated_summary

# Writing JSON data to file with indentation
with open("../../data/rl.json", 'w') as outfile:
    json.dump(d, outfile, indent=4)
