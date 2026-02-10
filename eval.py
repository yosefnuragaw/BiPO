import argparse
import os
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass, field
from typing import Tuple, Dict, List, Optional
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
from types import SimpleNamespace
from tqdm import tqdm
from transformers import pipeline


from utils import set_seed, get_eval_data
from models import BlockWrapper

SYSTEM_PROMPT = "You are a helpful, honest and concise assistant."

@dataclass
class ScriptArguments:
    """
    The arguments for the DPO eval script, matching the training config structure.
    """
    model_name_or_path: Optional[str] = field(
        default="google/gemma-3-1b-it",
        metadata={"help": "The model checkpoint for weights initialization."}
    )
    behavior: Optional[str] = field(default="power-seeking", metadata={"help": "the behavior"})
    layer: Optional[List[int]] = field(
        default_factory=lambda: list(range(26)), 
        metadata={"help": "the layer the steering vector extracted from"}
    )

    vec_dir: Optional[str] = field(
        default="/kaggle/working/BiPO/vector/power-seeking_gemma-3",
        metadata={"help": "Directory where .pt vectors are saved"}
    )
    eval_epoch: Optional[int] = field(default=18, metadata={"help": "Which epoch's vector to load"})

    prompt: Optional[str] = field(default="", metadata={"help": "What prompts for generation eval"})

class MultipleOptionDataset(Dataset):
    def __init__(self, tokenizer, prompts: List[List[str]], questions: List[str], labels: List[str]):
        super().__init__()
        self.tokenizer = tokenizer
        self.prompts = prompts
        self.questions = questions
        self.labels = labels

    def __getitem__(self, index: int):
        context_str = self.questions[index]

        tokenized_row = []
        for p in self.prompts[index]:
            full_text = context_str + " " + str(p)
            tok = self.tokenizer(full_text, 
                                 return_tensors='pt', 
                                 add_special_tokens=False)
            tokenized_row.append(tok)
        
        tokenized_question = self.tokenizer(context_str, 
                                            return_tensors='pt', 
                                            add_special_tokens=False)

        return {
            "question_length": tokenized_question.input_ids.shape[1],
            "input_ids": [tok.input_ids.squeeze(0) for tok in tokenized_row],
            "attention_mask": [tok.attention_mask.squeeze(0) for tok in tokenized_row],
            "label": self.labels[index],
        }

    def __len__(self) -> int:
        return len(self.questions)


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


def eval_accuracy(model, loader: DataLoader, multiplier: float, layers: List[int], epoch: int, vec_dir: str, verbose: bool = False) -> float:
    OPT = ['A', 'B']
    correct = 0
    total = 0
    
    if verbose:
        pbar = tqdm(loader, desc="Evaluating", ncols=100)
    else:
        pbar = loader
    
    for batch in pbar:
        label = batch["label"][0]
        q_len = batch["question_length"]
    
        for layer in layers:
            if isinstance(model.model.layers[layer], BlockWrapper):
                if label != 'A':
                    model.model.layers[layer].set_multiplier(-multiplier)
                else:
                    model.model.layers[layer].set_multiplier(multiplier)

        avg_logp = []
        for input_ids, attention_mask in zip(batch["input_ids"], batch["attention_mask"]):
            
            input_ids = input_ids.to(model.device)
            attention_mask = attention_mask.to(model.device)
    
            with torch.no_grad():
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                logps, _ = batch_logps(logits, input_ids)
                
                sliced = logps[0, q_len - 1:]
                avg_logp.append(sliced.mean().item())

        pred = OPT[avg_logp.index(max(avg_logp))]

        
        total += 1
        if pred == label:
            correct += 1
    
        current_acc = correct / total
        if verbose:
            pbar.set_description(f"Evaluating | Acc={current_acc:.4f}")

    return correct / total

def eval_generation(
    model,
    tokenizer,
    layers: list,
    multipliers: list,
    messages: list,
    device: int = 0,
    max_new_tokens: int = 32,
    temperature: float = 0.9,
):
    """
    Run generation for different steering multipliers on selected layers.
    """

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device if torch.cuda.is_available() else -1,
    )

    # --- Build chat prompt ---
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    results = {}
    for mult in multipliers:
        for layer in layers:
            model.model.layers[layer].set_multiplier(mult)

        output = generator(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
        )[0]["generated_text"]

        if "model" in output:
            trimmed = output[output.find("model") + 5:].strip()
        else:
            trimmed = output

        print(f"[Multiplier {mult}] : {trimmed}\n")
        results[mult] = trimmed

    return results

# --- Main Execution ---
if __name__ == "__main__":
   

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, required=True, help="Path to your YAML config file")
    parser.add_argument("--verbose", "-v", type=bool, required=False, default=True, help="Visualize eval progress")
    args, remaining = parser.parse_known_args()

    hf_parser = HfArgumentParser(ScriptArguments)
    if args.config.endswith(".yaml"):
        script_args = hf_parser.parse_yaml_file(yaml_file=args.config, allow_extra_keys=True)[0]
    elif args.config.endswith(".json"):
        script_args = hf_parser.parse_json_file(json_file=args.config, allow_extra_keys=True)[0]
    else:
        raise ValueError("Config file must be .yaml or .json")

    set_seed(seed=11)
  

    
    data = get_eval_data(script_args.behavior)
    
    print("Loading model to GPU...")
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    model.warnings_issued = {}
    model.config.use_cache = False
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    for layer in script_args.layer:
        vec_path = f"{script_args.vec_dir}/vec_ep{script_args.eval_epoch}_layer{layer}.pt"
        if os.path.exists(vec_path):
            layer_device = next(model.model.layers[layer].parameters()).device
            steering_vector = torch.load(vec_path, map_location=layer_device)
            
            model.model.layers[layer] = BlockWrapper(
                model.model.layers[layer], 
                hidden_dim=model.config.hidden_size, 
                vec=steering_vector
            )
            print(f"Loaded steering vector: {vec_path} on device {layer_device}")
        else:
            print(f"Warning: Vector not found at {vec_path}, skipping layer {layer}")

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    eval_dataset = MultipleOptionDataset(
        tokenizer=tokenizer,
        questions=data.questions,
        prompts=data.prompts,
        labels=data.labels,
    )
        
    eval_loader = DataLoader(
        dataset=eval_dataset,
        batch_size=1,              
        shuffle=False,          
        num_workers=0            
    )
    
    model.eval()
    print(f"Loaded config from {args.config}")
    # --- Accuracy Eval ---
    for mul in [0,1.,1.5,2,2.5,3]:
        
        accuracy = eval_accuracy(
            model=model,
            loader=eval_loader,
            multiplier=mul,
            layers=script_args.layer, 
            epoch=script_args.eval_epoch,
            vec_dir=script_args.vec_dir, 
            verbose=args.verbose
        )
        print(f"[Behavior:] {script_args.behavior} | [Epoch:] {script_args.eval_epoch} | [Multiplier:] {mul} | [Accuracy:] {accuracy:.4f}")

    # --- Generation eval ---
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": script_args.prompt},
    ]

    res = eval_generation(
        model=model,
        tokenizer=tokenizer,
        layers=script_args.layer,
        multipliers=[-3, -2, -1, 0, 1, 2, 3],
        messages=messages,
    )