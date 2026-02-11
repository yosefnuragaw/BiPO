from BiPO.models import MultipleOptionDataset
from BiPO.utils import get_eval_data
import torch
from collections import defaultdict
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.utils.data as data
from torch.utils.data import  DataLoader


def compute_fisher_trace(model, dataloader):
    """
    Computes the Empirical Fisher Trace
    """
    layer_fisher_traces = defaultdict(float)
    num_samples = 0
        
    for batch in tqdm(dataloader, desc="Computing Gradients"):
        for input_ids, attention_mask in zip(batch["input_ids"], batch["attention_mask"]):
            
            input_ids = input_ids.to(model.device)
            attention_mask = attention_mask.to(model.device)
        
            model.zero_grad()
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = outputs.loss
            loss.backward()
            
            for name, param in model.named_parameters():
                if param.grad is not None:       
                    parts = name.split('.')
                    
                    if "layers" in parts:
                        try:
                            idx = parts[parts.index("layers") + 1]
                            layer_key = int(idx)
                        except IndexError:
                            layer_key = "unknown"
                    elif "embed_tokens" in name:
                        layer_key = "embeddings"
                    elif "norm" in parts and "layers" not in parts:
                        layer_key = "final_norm"
                    elif "lm_head" in name:
                        layer_key = "lm_head"
                    else:
                        layer_key = "other"

                    
                    if  isinstance(layer_key,int):
                        layer_fisher_traces[layer_key] += torch.sum(param.grad ** 2).item()
            
        num_samples += 1

    avg_layer_traces = {k: v / num_samples for k, v in layer_fisher_traces.items()}
    return avg_layer_traces



if __name__ == '__main__':
    model_id = "google/gemma-3-1b-it" 
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        dtype=torch.bfloat16,
        device_map="auto" 
    )

    model.eval()
    
    ds = get_eval_data("power-seeking")
    eval_dataset = MultipleOptionDataset(
        tokenizer=tokenizer,
        questions=data.questions,
        prompts=data.prompts,
        labels=data.labels,
    )
        
    dataloader = DataLoader(
        dataset=eval_dataset,
        batch_size=1,              
        shuffle=False,          
        num_workers=0            
    )
    

    traces = compute_fisher_trace(model, dataloader, device=model.device)
    sorted_layers = sorted(traces.items(), key=lambda x: x[0])

    for layer, trace in sorted_layers:
        print(f"{layer:<15} | {trace:.4e}")