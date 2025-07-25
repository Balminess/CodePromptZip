**Replication Package for Paper Under Review**

This repository provides the replication package for our paper on compressing prompts of RAG-based coding task.

**Dataset Links:**

* Assertion Generation: [https://sites.google.com/view/atlas-nmt/home](https://sites.google.com/view/atlas-nmt/home)
* BugsFix: [https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/code-refinement/data/medium]
* Code Suggestion: [https://github.com/iCSawyer/CodeSuggestion?tab=readme-ov-file] lucene_java_header_code_tokens

**Compression Training Set Links:**
* Google Drive: [https://drive.google.com/file/d/1tGMP0nbMe1BgKxGi1M0Qj2_HgflFPCUB/view?usp=share_link]
  
**Hugginggce Link:**
* zip: [https://huggingface.co/Balminess/zipt5]

  
**Framework:**
```
CodePromptZip
  └──  Program Analysis    # JavaParser-based method to generate compressed code at varying compression ratios, constructing diverse code compression datasets.
       └── finetuning_data # Stores datasets used for fine-tuning models on compressed code representations.
  └──  Retrieval
       ├── preprocess.py   # Uses the Sparse Retriever to retrieve code examples and their compressed versions.
       └── retrieved_examples/ # JSONL files storing retrieved examples under the key demonstration.'demonstration'
  └──  Construction
       ├── construction.py # Uses retrieved examples to construct retrieval-augmented generation (RAG) prompts with predefined templates.
       └── prompt/         # JSONL files containing prompts with compressed or original code examples for the base LLM.
  └──  Generation
       ├── generation.py   # Sends requests to the base LLM to obtain responses.
       ├── evaluation.py   # Evaluates the quality of generated results.
       └── results         # Stores generated results for analysis.
  └──  CodeT5Copy
       ├── model.py  
       └── finetuning.py   # Fine-tunes CodeT5Copy using the constructed code compression datasets.    
```  
