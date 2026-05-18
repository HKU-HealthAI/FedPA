import os
import json
import shutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--cache_dir", default="./cache_ours", type=str, help="Cache directory for the model.")
parser.add_argument("--alpha_helpfulness", type=float, default=0.5, help='M = M_base + alpha_helpfulness * M_helpfulness + alpha_harmlessness * M_harmlessness')
parser.add_argument("--alpha_harmlessness", type=float, default=0.5, help='M = M_base + alpha_helpfulness * M_helpfulness + alpha_harmlessness * M_harmlessness')
parser.add_argument("--model_base_name_or_path", default="/path/to/model_cache/model--PKU-Alignment--alpaca-7b-reproduced")
parser.add_argument("--model_fedpa_both_name_or_path", default="../training/exp_dir/final_checkpoint")
parser.add_argument("--save_path", default="/path/to/model_cache/merged_model")
args = parser.parse_args()


##### change pref_vec_init for reward ####
model_name = f'FedPA_{args.alpha_helpfulness}help_{args.alpha_harmlessness}harm'
cache_path = os.path.join(args.cache_dir, model_name)
os.makedirs(cache_path, exist_ok=True)
with open(f'{args.model_fedpa_both_name_or_path}/adapter_config.json', 'r') as f:
    config = json.load(f)

config['pref_vec_init'] = [args.alpha_harmlessness, args.alpha_helpfulness]

with open(f'{cache_path}/adapter_config.json', 'w', encoding='utf-8') as f:
    json.dump(config, f, ensure_ascii=False, indent=4)

shutil.copyfile(f'{args.model_fedpa_both_name_or_path}/adapter_model.safetensors', f'{cache_path}/adapter_model.safetensors')

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(args.model_base_name_or_path, torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True)
print("Loading LoRA adapter...")
base_model_with_lora = PeftModel.from_pretrained(base_model, cache_path)    # notice here! use the cache_path instead of model_fedpa_both_name_or_path
print("Merging LoRA weights into base model...")
merged_model = base_model_with_lora.merge_and_unload()
print(f"Saving merged model to {args.save_path}...")
merged_model.save_pretrained(args.save_path)

tokenizer = AutoTokenizer.from_pretrained(args.model_base_name_or_path)
tokenizer.save_pretrained(args.save_path)
print("Merge and save completed successfully!")