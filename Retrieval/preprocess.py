import os
import jsonlines
import csv
import numpy as np
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModel, RobertaModel
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.neighbors import BallTree
import faiss
import time
from annoy import AnnoyIndex
from rank_bm25 import BM25Okapi,BM25L

def write_to_files(processed,output_path,file_name):
    with jsonlines.open(os.path.join(output_path, file_name),'w') as f:
        f.write_all(processed)


def rankbm25_preprocess(kdbase,zip_question_1,zip_question_2,test,path,numbers,ratio):
    
    tokenized_corpus = [doc.split(" ") for doc in question]
    bm25 = BM25Okapi(tokenized_corpus) 
    
    processed = []
    start_time = time.perf_counter()
    for obj in tqdm(test, total=len(test)):
        query = (obj['focal_method']+obj['test_method']).split(" ")
        score = bm25.get_scores(query)
        rtn =np.argsort(score)[::-1][:number] 
        code_candidates_tokens = []
        for i in rtn:
            code_candidates_tokens.append({'focal_method':zip_question_1[i],'test_method': zip_question_2[i],'assertion': answer[i],'idx':int(i)})        
        processed.append({'focal_method': obj['focal_method'],'test_method': obj['test_method'],'assertion': obj['assertion'],'demonstrations': code_candidates_tokens})
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"bmExecution time: {execution_time:.6f} seconds")
    
    write_to_files(processed,output_path,f'atlas_rankbm25_{ratio}.jsonl')

def sbert_preprocess(kdbase,zip_question_1,zip_question_2,test,path,numbers,ratio):
    model = SentenceTransformer("flax-sentence-embeddings/st-codesearch-distilroberta-base")
    kd_embedding = model.encode(kdbase)

    processed = []
    start_time = time.perf_counter()
    for obj in tqdm(test):
        query = obj['focal_method']+obj['test_method']
        query_emb = model.encode(query, convert_to_tensor=True)

        hits = util.semantic_search(query_emb, kd_embedding, top_k=number)[0]
        code_candidates_tokens = []
        for i in range(len(hits)):
            code_candidates_tokens.append({question_key_1: zip_question_1[hits[i]['corpus_id']],question_key_2: zip_question_2[hits[i]['corpus_id']],answer_key: answer[hits[i]['corpus_id']], 'idx':int(hits[i]['corpus_id'])})
        processed.append({question_key_1: obj[question_key_1],question_key_2: obj[question_key_2],answer_key: obj[answer_key], 'code_candidates_tokens': code_candidates_tokens})
    
    end_time = time.perf_counter()
    execution_time = end_time - start_time

    print(f"sbExecution time: {execution_time:.6f} seconds")
    write_to_files(processed,output_path,f'atlas_sbert_{ratio}.jsonl')


if __name__ == "__main__": 
    ratio=0.5
    output_path='/preprocess'
    train_path='atlas_codezip_0.jsonl'
    compressed_kd_base=f'atlas_codezip_{ratio}.jsonl'
    test_path='atlas-test-m.jsonl'
    
    kdbase= []

    with jsonlines.open(train_path) as f:   
        for i in f:
            kdbase.append(i)

    zip_kdbase=[]
    with jsonlines.open(compressed_kd_base) as f:   
        for i in f:
            kdbase.append(i)

    test = []
    with jsonlines.open(test_path) as f:
        for i in f:
            test.append(i)

    question_key_1 ='focal_method'
    question_key_2 ='test_method'
    answer_key ='assertion'

    question = [obj['focal_method']+obj['test_method']for obj in kdbase] 
    zip_question_1=[obj['focal_method'] for obj in zip_kdbase] 
    zip_question_2=[obj['test_method']for obj in zip_kdbase] 
    answer = [obj[answer_key] for obj in kdbase]

    numbers=64

    rankbm25_preprocess(kdbase,zip_question_1,zip_question_2,test,output_path, numbers,ratio) 
    sbert_preprocess(kdbase,zip_question_1,zip_question_2,test,output_path, numbers,ratio)  


