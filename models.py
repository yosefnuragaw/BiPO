from typing import List
import torch
from torch.utils.data import Dataset
from fastchat.conversation import Conversation

class BlockWrapper(torch.nn.Module):
    def __init__(self, block, hidden_dim, vec=None):
        super().__init__()
        self.multiplier = 1.0
        self.block = block

        try:
            ref_param = next(block.parameters())
            init_dtype = ref_param.dtype
        except StopIteration:
            init_dtype = torch.float32
            
        if vec is not None:
            self.vec = vec
        else:
            self.vec = torch.nn.Parameter(torch.zeros(hidden_dim, dtype=init_dtype))

    def forward(self, *args, **kwargs):
        output = self.block(*args, **kwargs)
        if isinstance(output, tuple):
            modified_hidden = output[0] + (self.multiplier * self.vec)
            return (modified_hidden,) + output[1:]
        elif isinstance(output, torch.Tensor):
            return output + (self.multiplier * self.vec)
        else:
            return output

    def set_multiplier(self, multiplier):
        self.multiplier = multiplier

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.block, name)
        
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
            full_text = context_str + str(p)+"<end_of_turn>\n"
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
        return len(self.prompts)

