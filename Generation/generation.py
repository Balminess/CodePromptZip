import os
import tqdm
import json
import pandas as pd
import openai
import numpy as np
import jsonlines
import litellm

os.environ["OPENAI_API_KEY"] = 'openai'

def unified_inference(prompt_path, result_path, model_name="gpt-3.5-turbo"):
    max_retries = 5
    queries = []
    
    with jsonlines.open(prompt_path) as reader:
        for obj in reader:
            queries.append(obj)
    
    data = []
    for pos, query in enumerate(queries):
        print(pos)
        attempts = 0
        while attempts < max_retries:
            try:
                response_actual = litellm.completion(
                    model=model_name,
                    messages=[{"role": "user", "content": query['prompt']}]
                )
                
                result = {
                    'label': query['label'],
                    'actual': response_actual["choices"][0]["message"]["content"],
                    'idx': pos
                }
                data.append(result)
                break  
            except Exception as e:
                attempts += 1
                print(f"Attempt {attempts} failed with error: {e}")
                if attempts == max_retries:
                    print("Max retries reached. Moving to the next item.")
                    break  
    
    with jsonlines.open(result_path, mode='w') as f:
        f.write_all(data)1

if __name__ == '__main__':
    unified_inference("input.jsonl", "output.jsonl", model_name="gpt-3.5-turbo")
