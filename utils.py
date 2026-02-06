
import random
import numpy as np
import torch
from typing import Dict
from datasets import load_dataset
from fastchat.conversation import get_conv_template
import os
from types import SimpleNamespace

SYSTEM_PROMPT = "You are a helpful, honest and concise assistant."

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.4f}"
    )

def get_data(num_proc=1, behavior='power-seeking', train=True, template_name='llama-2'):
    file_path = f"./data/{behavior}/{'train' if train else 'test'}.csv"
    dataset = load_dataset("csv", data_files=file_path, split='train')
    original_columns = dataset.column_names
    
    def return_prompt_and_responses(samples) -> Dict[str, str]:
        prompt = []
        for question in samples["question"]:
            conv = get_conv_template(template_name)
            conv.set_system_message(SYSTEM_PROMPT)
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            prompt.append(conv.get_prompt())
        return {
            "prompt": prompt,
            "chosen": [' ' + s for s in samples["matching"]],
            "rejected": [' ' + s for s in samples["not_matching"]],
        }

    return dataset.map(
        return_prompt_and_responses,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )

def get_eval_data(behavior):
    # Ensure path exists or handle error
    path = f"./data/{behavior}/test_infer.csv"
    if not os.path.exists(path):
         raise FileNotFoundError(f"Data file not found: {path}")
         
    dataset = load_dataset("csv", data_files=path, split='train')
    questions = []
    labels = []
    prompts = []
    
    for row in dataset:
        P = (f"{SYSTEM_PROMPT}.\n{row['question']}\n\nAnswer:")
        questions.append({"role": "user", "content": P})
        
        # Extract options dynamically
        prompts.append([row[col] for col in dataset.column_names if col in ['A','B','C','D']])
        labels.append(row['matching'])

    return SimpleNamespace(
        questions = questions,
        prompts = prompts,
        labels = labels,
    )