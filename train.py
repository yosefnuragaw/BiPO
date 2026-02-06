import argparse
from dataclasses import dataclass, field
from typing import List, Optional
import os
import wandb


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
from trl import BiPOTrainer, DPOConfig
from fastchat.conversation import Conversation, conv_templates

from utils import get_data, print_trainable_parameters, set_seed
from models import BlockWrapper


MODEL_TEMPLATE_MAP = {
    'meta-llama/Llama-2-7b-chat-hf': 'llama-2',
    'mistralai/Mistral-7B-Instruct-v0.2': 'mistral',
    'google/gemma-3-1b-it': 'gemma-3'
}


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

# --- Arguments ---
@dataclass
class ScriptArguments:
    beta: Optional[float] = field(default=0.05, metadata={"help": "the beta parameter for DPO loss"})
    model_name_or_path: Optional[str] = field(
        default="google/gemma-3-1b-it",
        metadata={"help": "Supported: meta-llama/Llama-2-7b-chat-hf, mistralai/Mistral-7B-Instruct-v0.2, google/gemma-3-1b-it"},
    )
    learning_rate: Optional[float] = field(default=1e-2, metadata={"help": "optimizer learning rate"})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "the lr scheduler type"})
    warmup_steps: Optional[int] = field(default=10, metadata={"help": "the number of warmup steps"})
    weight_decay: Optional[float] = field(default=0.001, metadata={"help": "the weight decay"})
    optimizer_type: Optional[str] = field(default="adamw_torch", metadata={"help": "the optimizer type"})

    per_device_train_batch_size: Optional[int] = field(default=4, metadata={"help": "train batch size per device"})
    per_device_eval_batch_size: Optional[int] = field(default=1, metadata={"help": "eval batch size per device"})
    gradient_accumulation_steps: Optional[int] = field(default=1, metadata={"help": "gradient accumulation steps"})
    gradient_checkpointing: Optional[bool] = field(default=False, metadata={"help": "use gradient checkpointing"})

    max_prompt_length: Optional[int] = field(default=2048, metadata={"help": "maximum prompt length"})
    max_length: Optional[int] = field(default=2048, metadata={"help": "maximum sequence length"})
    num_train_epochs: Optional[int] = field(default=20, metadata={"help": "number of training epochs"})
    logging_steps: Optional[int] = field(default=1, metadata={"help": "logging frequency"})
    log_freq: Optional[int] = field(default=1, metadata={"help": "logging frequency"})

    behavior: Optional[str] = field(default="power-seeking", metadata={"help": "the behavior"})
    layer: Optional[List[int]] = field(
        default_factory=lambda: list(range(26)), 
        metadata={"help": "the layer the steering vector extracted from"}
    )
    report_to: Optional[str] = field(default="wandb", metadata={"help": "integration to report to"})
    ignore_bias_buffers: Optional[bool] = field(default=False, metadata={"help": "fix for DDP issues"})





# --- Main Execution ---
if __name__ == "__main__":
    # 1. Parse Args (YAML support)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to your YAML config file")
    args, remaining = parser.parse_known_args()

    hf_parser = HfArgumentParser(ScriptArguments)
    if args.config.endswith(".yaml"):
        script_args = hf_parser.parse_yaml_file(yaml_file=args.config)[0]
    elif args.config.endswith(".json"):
        script_args = hf_parser.parse_json_file(json_file=args.config)[0]
    else:
        raise ValueError("Config file must be .yaml or .json")

    layer_str = "-".join(map(str, script_args.layer))
    run_name = f"{script_args.behavior}-Layers_{layer_str}"

    os.environ["WANDB_NAME"] = run_name
    set_seed(seed=11)
    
    # 2. Determine Template Name
    if script_args.model_name_or_path not in MODEL_TEMPLATE_MAP:
        print(f"Warning: {script_args.model_name_or_path} not in supported list: {list(MODEL_TEMPLATE_MAP.keys())}")
    template_name = MODEL_TEMPLATE_MAP.get(script_args.model_name_or_path, 'llama-2')

    print(f"Loaded config from {args.config}")
    print(f"[Behavior:] {script_args.behavior} | [Layer:] {script_args.layer} | [Model:] {script_args.model_name_or_path}")

    # 3. Load & Configure Models
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    model.warnings_issued = {}
    model.config.use_cache = False

    # Inject BlockWrappers
    for layer in script_args.layer:
        model.model.layers[layer] = BlockWrapper(model.model.layers[layer], hidden_dim=model.config.hidden_size)

    if script_args.ignore_bias_buffers:
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    # Load Reference Model
    model_ref = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    # 4. Freeze/Unfreeze Logic
    print('Freezing base model parameters...') 
    for param in model_ref.parameters():
        param.requires_grad = False
    for param in model.parameters():
        param.requires_grad = False

    print('Unfreezing steering vectors...')
    for layer in script_args.layer:
        model.model.layers[layer].vec.requires_grad = True  

    # 5. Load Datasets
    train_dataset = get_data(behavior=script_args.behavior, train=True, template_name=template_name) 
    test_dataset = get_data(behavior=script_args.behavior, train=False, template_name=template_name) 
    
    # 6. Initialize Training Args
    training_args = DPOConfig(
        output_dir="placeholder", # required by TRL but not used in this specific flow
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        num_train_epochs=script_args.num_train_epochs,
        logging_steps=script_args.logging_steps,
        save_strategy="no",
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        gradient_checkpointing=script_args.gradient_checkpointing,
        learning_rate=script_args.learning_rate,
        eval_strategy="epoch",
        report_to=script_args.report_to,
        lr_scheduler_type=script_args.lr_scheduler_type,
        warmup_steps=script_args.warmup_steps,
        optim=script_args.optimizer_type,
        bf16=False,
        remove_unused_columns=False,
        max_prompt_length=script_args.max_prompt_length,
        max_length=script_args.max_length,
        beta=script_args.beta,
    )

    # 7. Start Trainer
    dpo_trainer = BiPOTrainer(
        model=model,
        ref_model=model_ref,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset={'test_dataset_add': test_dataset, 'test_dataset_sub': test_dataset},
        processing_class=tokenizer,
        behavior=script_args.behavior,
        layer=script_args.layer,
        name=template_name,
    )

    print_trainable_parameters(model)
    dpo_trainer.train()