import os
import re
import jsonlines
import torch
import torch.nn as nn
from transformers import RobertaTokenizer
from tqdm import tqdm
from model import T5ForConditionalGenerationWithCopyMech
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = RobertaTokenizer.from_pretrained('tokenizer_path')
special_tokens = ["<Assertion>", "<Bugs2fix>", "<Suggestion>",'<ratio>','</ratio>','<compressed>','</compressed>']
tokenizer.add_tokens(special_tokens)

model = T5ForConditionalGenerationCopyMech.from_pretrained('model_path')
model.to(device)


max_length = 200  
eos_token_id = tokenizer.eos_token_id  

def extract_code(code, index=0):
    """
    Extract the content between <compressed></compressed> tags from a string.
    """
    pattern = r'<compressed>(.*?)</compressed>'
    matches = re.findall(pattern, code)
    if 0 <= index < len(matches):
        return matches[index]
    return None

def compress(org_code):
    """
    Compress code using the T5 model.
    """

    input_ids = tokenizer(org_code, return_tensors="pt").input_ids.to(device)
    decoder_input_ids = torch.tensor([[tokenizer.pad_token_id]], device=device)

    output_ids = decoder_input_ids.clone() 

    for _ in range(max_length):
        outputs = model(input_ids=input_ids, decoder_input_ids=output_ids)
        logits = outputs.logits
        next_token_logits = logits[:, -1, :]  
        next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)  
        output_ids = torch.cat([output_ids, next_token_id], dim=-1) 

        if next_token_id.item() == eos_token_id:
            break

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

def prompt_construction(shot_number, compress_flag, ratio):
    """
    Construct prompts for fine-tuning or evaluation.
    """

    number = shot_number  # 
    preprocess_file ='retrieval_path'
    output_path = 'prompt_path'

    test = []
    with jsonlines.open(preprocess_file) as f:
        for i in f:
            test.append(i)

    prompts = []

    for obj in test:
        topk = obj['demonstrations'][:number] if number else []
        demons = ""

        if not compress_flag:  
            for sample in reversed(topk):
                demons += '### FOCAL METHOD\n'
                demons += sample['focal_method'].strip() + '\n'
                demons += '### UNIT TEST\n'
                demons += sample['test_method'].strip() + '\n'
                demons += '### ASSERTION\n'
                demons += sample['assertion'].strip() + '\n'
        else:  
            for sample in reversed(topk):
          
                org_code = f"[Assertion]<ratio>{ratio}</ratio><compressed>{sample['focal_method']}</compressed><compressed>{sample['test_method']}</compressed>"
                compressed_code = compress(org_code).strip()

                demons += '### FOCAL METHOD\n'
                demons += extract_code(compressed_code, 0).strip() + '\n'
                demons += '### UNIT TEST\n'
                demons += extract_code(compressed_code, 1).strip() + '\n'
                demons += '### ASSERTION\n'
                demons += sample['assertion'].strip() + '\n'

 
        prompt = demons + '### FOCAL METHOD\n'
        prompt += obj['focal_method'] + '\n'
        prompt += '### UNIT TEST\n'
        prompt += obj['test_method'] + '\n'
        prompt += '### ASSERTION\n'

        prompts.append({'prompt': prompt, 'label': obj['assertion']})

    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, f'assertion_{number}_shot.jsonl')
    with jsonlines.open(output_file, 'w') as f:
        f.write_all(prompts)

if __name__ == "__main__":
    prompt_construction(shot_number=1, compress_flag=True, ratio=0.5)
