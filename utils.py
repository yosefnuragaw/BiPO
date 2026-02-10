
import random
import numpy as np
import torch
from typing import Dict
from datasets import load_dataset
from fastchat.conversation import get_conv_template
import os
from types import SimpleNamespace
from fastchat.conversation import Conversation, conv_templates

SYSTEM_PROMPT = "You are a helpful, honest and concise assistant."

class Gemma3Conversation(Conversation):
    def __init__(self):
        super().__init__(
            name="gemma-3",
            system_template="<bos><start_of_turn>system\n{system_message}<end_of_turn>\n",
            roles=("user", "assistant"),
            messages=[],
            sep="",
            sep2="",
            stop_str="<end_of_turn>",
            stop_token_ids=[1],  # Gemma EOS
        )

    def append_message(self, role, message):
        if role == "user":
            formatted = f"<start_of_turn>user\n{message}<end_of_turn>\n"
            self.messages.append((role, formatted))
        elif role == "assistant":
            formatted = f"<start_of_turn>model\n{message}<end_of_turn>\n"
            self.messages.append((role, formatted))
        else:
            raise ValueError(f"Unknown role: {role}")

    def get_prompt(self):
        prompt = ""
        if self.system_message:
            prompt += self.system_template.format(system_message=self.system_message)
        
        for _, content in self.messages:
            prompt += content
            
        prompt += "<start_of_turn>model\n"
        return prompt


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
        conv.append_message(conv.roles[0], row['question'])
        conv.append_message(conv.roles[1], None)
        
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