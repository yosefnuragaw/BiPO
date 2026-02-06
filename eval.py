import torch
from typing import Tuple, Dict, List, Optional
from torch.utils.data import Dataset,DataLoader
import pandas as pd
from datasets import Dataset, load_dataset
from fastchat.conversation import get_conv_template
from types import SimpleNamespace

SYSTEM_PROMPT = "You are a helpful, honest and concise assistant."

class BlockWrapper(torch.nn.Module):
    def __init__(self, block, hidden_dim ,vec=None):
        super().__init__()
        self.multiplier = 1.0
        self.block = block
        self.hidden_dim = hidden_dim

        try:
            self.ref_param = next(block.parameters())
            self.init_dtype = self.ref_param.dtype
        except StopIteration:
            self.init_dtype = torch.float32
            
        if vec is not None:
            self.vec = torch.nn.Parameter(vec)
        else:
            # Zero Init
            self.vec = torch.nn.Parameter(torch.zeros(self.hidden_dim,dtype=self.init_dtype))

    def forward(self, *args, **kwargs):
        output = self.block(*args, **kwargs)
        if isinstance(output, tuple):
            modified_hidden = output[0] + (self.multiplier * self.vec)
            return (modified_hidden,) + output[1:]
        
        elif isinstance(output, torch.Tensor):
            # Case B: Output is a direct Tensor (e.g., Gemma 3 sometimes does this)
            # Apply vector directly
            return output + (self.multiplier * self.vec)
            
        else:
            # Fallback (shouldn't happen, but safe to have)
            return output

    def set_multiplier(self, multiplier):
        self.multiplier  = multiplier

    def __getattr__(self, name):
            """
            Forward missing attributes (like 'attention_type') to the wrapped block.
            """
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.block, name)
            
    def detach_vec(self):
        self.vec = torch.nn.Parameter(torch.zeros(self.hidden_dim,dtype=self.init_dtype))


class MultipleOptionDataset(Dataset):
    def __init__(
            self,
            tokenizer,
            prompts:List[str],
            questions:List[str],
            labels:List[str],
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.prompts = prompts
        self.questions = questions
        self.labels = labels

    def __getitem__(self, index:int):
        context_str = self.tokenizer.apply_chat_template(self.questions[index], add_generation_prompt=True, tokenize=False)

        tokenized_row = [self.tokenizer(context_str+ " " + p, 
                                  return_tensors = 'pt',
                                  add_special_tokens=False)
                                  for p in self.prompts[index]]
        
        tokenized_question = self.tokenizer(context_str, 
                                  return_tensors = 'pt',
                                  add_special_tokens=False
        )


        return {
            "question_length": tokenized_question.input_ids.shape[1],
            "input_ids": [tok.input_ids for tok in tokenized_row],
            "attention_mask": [tok.attention_mask for tok in tokenized_row],
            "label": self.labels[index],
        }
    
    def __len__(self) -> int:
        return len(self.prompts)


def batch_logps(
        logits:torch.Tensor, 
        ids:torch.Tensor, 
        pad_id:int|None = None
        )->Tuple[torch.Tensor,torch.Tensor]:
        """
        Docstring for batch_logps
        
        :param logits: Description
        :type logits: torch.Tensor
        :param ids: Description
        :type ids: torch.Tensor
        :param pad_id: Description
        :type pad_id: int
        """

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

def eval(
        self, 
        model,
        loader:Dict,
        steering_vector: torch.Tensor|None = None,
        multiplier: float = 1.0 ,
        layers: List[int] = [15],
        )->float:
        
        preds = []
        OPT = ['A','B']

        for batch in loader:
            label = batch["label"].pop()
            communities = batch["communities"].pop()
            q_len = batch["question_length"]

            if steering_vector is None:
                steering_vector = torch.zeros(self.model.hidden_size).to(self.model.device)

            if label != 'A':
                steering_vector = -steering_vector

            for layer in layers:
                model.model.layers[layer] = BlockWrapper(model.model.layers[layer], hidden_dim=model.config.hidden_size, vec=multiplier*steering_vector)
                model.model.layers[layer].set_multiplier(multiplier)
            model.config.use_cache = False

            avg_logp= []
            for input_ids,attention_mask in zip(batch["input_ids"],batch["attention_mask"]):
                input_ids = input_ids.to(self.model.device).squeeze(0) 
                attention_mask = attention_mask.to(self.model.device).squeeze(0) 
    
                with torch.no_grad():
                    pos_logits = model(
                        input_ids=input_ids, 
                        attention_mask= attention_mask,
                    ).logits 
    
                    logps,_ = batch_logps(
                        logits=pos_logits, 
                        ids=input_ids)

                    sliced_logps = logps[0, q_len-1:]
                    avg_logp.append(sliced_logps.mean().item())

            
            pred = OPT[avg_logp.index(max(avg_logp))]
            preds.append((pred == label))

            for layer in layers:
                model.model.layers[layer].detach_vec()

        return sum(preds)/len(preds)

def get_data(behavior):
    dataset = load_dataset("csv", data_files=f"./data/{behavior}/test_infer.csv", split='train')

    questions = []
    labels = []
    prompts = []
    
    for row in dataset:
        P = (
                f"{SYSTEM_PROMPT}.\n"
                f"{row['question']}\n\n"
                f"Answer:"
            )

        questions.append({"role": "user", "content": P})
        prompts.append([row[col] for col in dataset.column_names if col in ['A','B','C','D']])
        labels.append(row['matching'])
        

    return SimpleNamespace(
        questions = questions,
        prompts = prompts,
        labels = labels,
    )

if __name__ == "__main__":
     


     
    data = get_data('power-seeking')
    eval_dataset = MultipleOptionDataset(
            tokenizer = model.tokenizer,
            questions = data.questions,
            prompts = data.prompts,
            labels = data.labels,
        )
        
    eval_loader = DataLoader(
            dataset = eval_dataset,
            batch_size=1,              
            shuffle=True,          
            num_workers=0           
            )
    
    accuracy = eval(eval_loader)
    print(f"{key} | Acc: {accuracy:3f}")
