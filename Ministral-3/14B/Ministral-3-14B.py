import sys
import os
import subprocess
import torch

def install_package(pkg_name:str):
    try:
        __import__(pkg_name)
        print(f'{pkg_name} already installed')
    except ImportError:
        subprocess.check_call(['uv', 'pip', 'install', pkg_name])
        print(f'{pkg_name} successfully installed')

pkgs = ['python-dotenv', 'transformers', 'trl', 'accelerate', 'bitsandbytes', 'datasets', 'ninja', 'peft', 'codecarbon', 'packaging', 'zeus', 'wandb', 'adapters', 'huggingface-hub', 'tqdm', 'evaluate', 'bert_score', 'google-tunix']

for pkg in pkgs:
    install_package(pkg)

from dotenv import load_dotenv
from huggingface_hub import login, auth_list
import wandb

load_dotenv()
wandb_key = os.getenv('WANDB')
hf_token = os.getenv('HF_TOKEN')

wandb.login(key=wandb_key)
login(token=hf_token, add_to_git_credential=False)

auth_list()

import json
from datetime import datetime
from datasets import load_dataset
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer
from codecarbon import EmissionsTracker
from codecarbon.output import LoggerOutput
import logging
from zeus.device import get_gpus
from zeus.monitor import ZeusMonitor

gpus = get_gpus()
print(gpus)

def format_sample(example):
    prompt = example["instruction"]
    if "input" in example and len(example["input"]) > 0:
        prompt += "\n" + example["input"]
    answer = example["output"]
    return {
        "text": f"<s>[INST]{prompt}[/INST]{answer}</s>"
    }


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = model.num_parameters()
    for _, param in model.named_parameters():
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


os.makedirs('code_carbon_ministral_3_14b', exist_ok=True)

log_name = "bioministral_3_14b_logs"
_logger = logging.getLogger(log_name)
_channel = logging.FileHandler(log_name + '.log')
_logger.addHandler(_channel)
_logger.setLevel(logging.INFO)

my_logger = LoggerOutput(_logger, logging.INFO)

tracker = EmissionsTracker(
    project_name = 'bio-ministral-3-14b',
    output_dir="./code_carbon_ministral_3_14b/",
    save_to_file=True,
    on_csv_write='append',
    output_file="emissions.csv",
    tracking_mode="process",
    measure_power_secs=1,
    save_to_logger=True,
    logging_logger=my_logger
)
tracker.start()
monitor = ZeusMonitor(gpu_indices=[torch.cuda.current_device()])

print('Mapping dataset')
monitor.begin_window('dataset')
dataset = load_dataset("bio-nlp-umass/bioinstruct", split="train")
dataset = dataset.map(format_sample)

ds_energy = monitor.end_window('dataset')

monitor.begin_window('load model')
model_name = "mistralai/Ministral-3-14B-Instruct-2512"
device = 'cuda'
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.pad_token = tokenizer.unk_token
tokenizer.pad_token_id = tokenizer.unk_token_id

compute_dtype = getattr(torch, "float16")


print('Downloading model')
model = AutoModelForCausalLM.from_pretrained(
          model_name,
          device_map="auto",
)

dataset_splits = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = dataset_splits['train']
test_dataset = dataset_splits['test']

peft_config = LoraConfig(
    lora_alpha=32,
    lora_dropout=0.07,
    r=32,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules= ['k_proj', 'q_proj', 'v_proj', 'o_proj', "gate_proj", "down_proj", "up_proj", "lm_head",]
)

model.config.pad_token_id = tokenizer.pad_token_id
model.config.use_cache = False

training_arguments = TrainingArguments(
    output_dir="./results",
    report_to="wandb",
    eval_strategy="epoch",
    optim="paged_adamw_8bit",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=1,
    log_level="info",
    save_steps=500,
    logging_steps=20,
    learning_rate=2e-5,
    num_train_epochs=1,
    warmup_steps=100,
    lr_scheduler_type="constant",
)

trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        args=training_arguments,
)
lmodel_energy = monitor.end_window('load model')

monitor.begin_window('fine-tuning')
trainer.train()

print_trainable_parameters(model)

new_model = 'Bio-ministral-3-14b'
trainer.model.save_pretrained(new_model)

ft_energy = monitor.end_window('fine-tuning')

emissions = tracker.stop()

import pandas as pd
import matplotlib.pyplot as plt

log_history = trainer.state.log_history
logs_df = pd.DataFrame(log_history)

train_loss = logs_df.dropna(subset=['loss'])
eval_loss = logs_df.dropna(subset=['eval_loss'])

plt.figure(figsize=(15, 8))
plt.plot(train_loss['epoch'], train_loss['loss'], label='Training Loss')
plt.plot(eval_loss['epoch'], eval_loss['eval_loss'], label='Validation Loss', marker='x')
plt.title('Función de pérdida')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()
plt.grid(True)
plt.show()
plt.tight_layout()
plt.savefig('loss_function.png', transparent=True)

print(f'Energy loading dataset: {ds_energy}')
print(f'Energy loading model: {lmodel_energy}')
print(f'Energy in fine-tuning model: {ft_energy}')