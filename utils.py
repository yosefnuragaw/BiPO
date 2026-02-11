
import random
import numpy as np
import torch
from typing import Dict,Tuple
from datasets import load_dataset
from fastchat.conversation import get_conv_template
import os
from types import SimpleNamespace
from fastchat.conversation import conv_templates

from models import Gemma3Conversation

SYSTEM_PROMPT = "You are a helpful, honest and concise assistant."
conv_templates["gemma-3"] = Gemma3Conversation()

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

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

def get_data(num_proc=1, behavior='power-seeking', train=True, template_name='gemma-3'):
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

def get_eval_data(behavior, template_name='gemma-3'):
    path = f"./data/{behavior}/test_infer.csv"
    if not os.path.exists(path):
         raise FileNotFoundError(f"Data file not found: {path}")
         
    dataset = load_dataset("csv", data_files=path, split='train')
    
    questions = [] 
    prompts = []  
    labels = []    
    
    for row in dataset:
        conv = get_conv_template(template_name)
        conv.set_system_message(SYSTEM_PROMPT)
        conv.append_message(conv.roles[0], f"{row['question']}")
        # conv.append_message(conv.roles[1], None)
        
        full_prompt = conv.get_prompt()
        questions.append(full_prompt)
        
        current_options = [row[col] for col in ['A','B','C','D'] if col in row]
        prompts.append(current_options)
        labels.append(row['matching'])

    return SimpleNamespace(
        questions=questions,
        prompts=prompts,
        labels=labels,
    )

def batch_logps(logits: torch.Tensor, ids: torch.Tensor, pad_id: int | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
    if logits.shape[:-1] != ids.shape:
        raise ValueError("Logits and ids must have the same shape. (batch,sequence_length,dim)")

    ids = ids.clone()
    ids = ids[:, 1:].contiguous()
    logits = logits[:, :-1, :].contiguous()

    loss_mask = None
    if pad_id is not None:
        loss_mask = ids != pad_id
        ids[ids == pad_id] = 0
        
    token_logps = torch.gather(logits.log_softmax(-1), dim=-1, index=ids.unsqueeze(-1)).squeeze(-1)
    return token_logps, loss_mask