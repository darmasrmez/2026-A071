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

pkgs = ['python-dotenv', 'transformers', 'trl', 'accelerate', 'bitsandbytes', 'datasets', 'ninja', 'peft', 'codecarbon', 'packaging', 'zeus', 'wandb', 'huggingface-hub', 'tqdm', 'evaluate', 'bert_score']


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


os.makedirs('code_carbon', exist_ok=True)

log_name = "biomistral_7b_logs"
_logger = logging.getLogger(log_name)
_channel = logging.FileHandler(log_name + '.log')
_logger.addHandler(_channel)
_logger.setLevel(logging.INFO)

my_logger = LoggerOutput(_logger, logging.INFO)

tracker = EmissionsTracker(
    project_name = 'bio-mistral-7b',
    output_dir="./code_carbon/",
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

tracker._geo
print('Mapping dataset')
monitor.begin_window('dataset')
dataset = load_dataset("bio-nlp-umass/bioinstruct", split="train")
dataset = dataset.map(format_sample)

ds_energy = monitor.end_window('dataset')

monitor.begin_window('load model')
model_name = "Hugofernandez/Mistral-7B-v0.1-colab-sharded"
device = 'cuda'
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.pad_token = tokenizer.unk_token
tokenizer.pad_token_id = tokenizer.unk_token_id

compute_dtype = getattr(torch, "float16")
bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
)

print('Downloading model')
model = AutoModelForCausalLM.from_pretrained(
          model_name,
          quantization_config=bnb_config,
          device_map="auto",
)

dataset_splits = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = dataset_splits['train']
test_dataset = dataset_splits['test']

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=16,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules= ['k_proj', 'q_proj', 'v_proj', 'o_proj', "gate_proj", "down_proj", "up_proj", "lm_head",]
)

model = prepare_model_for_kbit_training(model)
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

new_model = 'Biomistral_7B'
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
print(f'Energy in fine-tunit model: {ft_energy}')



# import evaluate
# from tqdm import tqdm

# bertscore = evaluate.load("bertscore")
# rouge = evaluate.load("rouge")


# predictions = []
# references = []

# print("Starting generation...")
# for row in tqdm(test_dataset):
#     prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

# ### Instruction:
# {row['instruction']}

# ### Input:
# {row['input']}

# ### Response:
# """

#     inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

#     with torch.no_grad():
#         outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.1)

#     response = tokenizer.decode(outputs[0], skip_special_tokens=True)

   
#     if "### Response:" in response:
#         response = response.split("### Response:")[-1].strip()

#     predictions.append(response)
#     references.append(row['output'])


# print("Calculating metrics...")
# bertscore_results = bertscore.compute(predictions=predictions, references=references, lang="en")
# print(f"BERTScore F1 (Mean): {sum(bertscore_results['f1']) / len(bertscore_results['f1'])}")

# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd

# energy_data = {
#     'Device': ['CPU', 'GPU', 'RAM', 'Total'],
#     'Energy (kWh)': [df['cpu_energy'].iloc[0], df['gpu_energy'].iloc[0], df['ram_energy'].iloc[0], df['energy_consumed'].iloc[0]]
# }
# plot_df = pd.DataFrame(energy_data)


# plt.figure(figsize=(10, 6))
# sns.barplot(x='Device', y='Energy (kWh)', hue='Device', data=plot_df, palette='viridis', legend=False)
# plt.title('Energía consumida por Dispositivo')
# plt.xlabel('Dispositivo de Hardware')
# plt.ylabel('Energía consumida en kWh')

# # Add the values on top of the bars
# for index, row in plot_df.iterrows():
#     plt.text(index, row['Energy (kWh)'], f"{row['Energy (kWh)']:.2f}", color='black', ha="center", va='bottom')

# plt.tight_layout()
# plt.savefig('energy_consumption.png', transparent=True)