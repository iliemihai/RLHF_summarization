import json 
from tqdm import tqdm 
from transformers import T5Tokenizer, T5ForConditionalGeneration 
 
 
d = json.load(open("../../data/rl.json")) 
 
model_name = "../sft/hf_model/" # "google/flan-t5-xl" 
tokenizer = T5Tokenizer.from_pretrained(model_name) 
model = T5ForConditionalGeneration.from_pretrained(model_name).to("cuda") 
h = open("saved_summaries_train.json", "w") 
h.write('{"texts": [') 
print("Generate for train...") 
rejected = [] 

for line, acc in tqdm(zip(d["x_train"], d["y_train"]), total=len(d["x_train"])): 
    prompt = "Summary: {0}".format(line) 
    input_ids = tokenizer(prompt, padding="max_length", max_length=512, truncation=True, return_tensors="pt").input_ids.to("cuda") 

    outputs = model.generate(input_ids, penalty_alpha=0.6, top_k=4, max_length=512) 
    out = tokenizer.decode(outputs[0])
    line = line.replace('\n', ' ').replace('"', ' ').replace("'", " ")
    print("ORIGINAL NEWS: ", line)
    print("====")
    print(out) 
    print("VS") 
    print(acc) 
    print("----------------------------------------------------") 
    rejected.append(out) 
    h.write('{ "prompt": "'+line+', "accepted": "'+acc.replace('\n', ' ').replace('"', ' ').replace("'", " ")+'", "rejected": "'+out.replace('\n', ' ').replace('"', ' ').replace("'", " ")+'"},\n') 
h.write(']}') 
h.close()

h = open("saved_summaries_valid.json", "w") 
print("Generate for validation...") 
rejected = [] 
for line in tqdm(d["x_val"]): 
    prompt = "Summary: {0}".format(line) 
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")  
    outputs = model.generate(input_ids, penalty_alpha=0.6, top_k=4, max_length=512) 
    out = tokenizer.decode(outputs[0])
    rejected.append(out) 
    rejected.append(out)
    line = line.replace('\n', ' ').replace('"', ' ').replace("'", " ")
    h.write('{ "prompt": "'+line+'{ "accepted": "'+acc.replace('\n', ' ').replace('"', ' ').replace("'", " ")+'", "rejected": "'+out.replace('\n', ' ').replace('"', ' ').replace("'", " ")+'"},\n')
    print("AVEM", out)
h.write(']}')
 
 
