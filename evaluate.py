import torch,os
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from peft import PeftModel
import pandas as pd
import numpy as np

lora_adapter=''
prompt='1'
dataset='Diagnosis'
model_size='7'
batch_size=2  
model_dir='../models/'
max_new_tokens=4096
    
base_model_path = f'{model_dir}/Qwen2.5-VL-{model_size}B-Instruct/'
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(base_model_path, torch_dtype="auto", device_map="auto")
processor = AutoProcessor.from_pretrained(base_model_path)
processor.tokenizer.padding_side = "left"

if(lora_adapter):
    lora_weights_path = f'saves/SFT_{model_size}B_{lora_adapter}/'
    model = PeftModel.from_pretrained(model, lora_weights_path)

lora=[]
for name, param in model.named_parameters():
    if "lora" in name:lora.append(name)
print("LoRA layers: ", len(lora))

def chat(messages):
    texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=texts,images=image_inputs,videos=video_inputs,padding=True,return_tensors="pt")
    inputs = inputs.to("cuda")
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_texts = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return output_texts

print('Base Model: ', base_model_path)
print('LORA: ', lora_weights_path if lora_adapter else 'NA')

df_file=f'data/{dataset}_test.csv'
results_file = f"results/{dataset}_{model_size}B_L{lora_adapter}_P{prompt}_HPC.txt"
df = pd.read_csv(df_file).fillna('')


df['images']=df['images'].apply(lambda x: eval(x))
if(prompt):
    with open(f'prompts/{dataset}-{prompt}.txt', 'r', encoding='utf8') as f: sys_prompt = f.read()

print('DF File: ', df_file)
print('Result File: ', results_file)
print(sys_prompt)

ds = []   
for batch_start in range(0, len(df), batch_size):
    batch = df[batch_start:batch_start + batch_size]
    batch_messages = []
    batch_rows = []
    
    for i, row in batch.iterrows():
        row = row.to_dict()
        content = []
        images = row['images']
        for image in images:
            img_content = {"type": "image", "image": 'data/' + image}
            content.append(img_content)
        content.append({'type': 'text', 'text': row['q']})
        messages = [{"role": "user", "content": content}]

        if(prompt): messages.insert(0,{"role": "system", "content": sys_prompt})
        batch_messages.append(messages)
        batch_rows.append(row)
    responses = chat(batch_messages)
    
    for idx, row in enumerate(batch_rows):
        new_row={'id': row['id'], 'true':row['a'], 'pred':responses[idx]}
        ds.append(new_row)
        for k, v in new_row.items(): print(f"{k}: {v}")
    pd.DataFrame(ds).to_csv(results_file, sep='\t', index=None)
del model

print('DONE!!')