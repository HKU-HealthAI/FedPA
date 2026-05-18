'''
The evaluation data is from the supplemetary material of the safe-RLHF paper: https://openreview.net/forum?id=TyFrPOKYXw
The template is for the Alpaca model in the safe-rlhf paper. Here we assume all the models (base and reward model) use the same template.
'''

import argparse
import json
from pathlib import Path
from tqdm import tqdm
import time
import os
from vllm import LLM, SamplingParams


PROMPT_BEGIN: str = 'BEGINNING OF CONVERSATION: '
PROMPT_USER: str = 'USER: {input} '
PROMPT_ASSISTANT: str = 'ASSISTANT:'  # should not have a space at the end
PROMPT_INPUT_ALPACA: str = PROMPT_BEGIN + PROMPT_USER + PROMPT_ASSISTANT

def str2bool(string: str) -> bool:
    """Convert a string literal to a boolean value."""
    if string.lower() in {'1', 'true', 't', 'yes', 'y', 'on'}:
        return True
    if string.lower() in {'0', 'false', 'f', 'no', 'n', 'off'}:
        return False
    return bool(string)

def parse_arguments() -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Generation of prompts using LLMs.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Model
    model_parser = parser.add_argument_group('model')
    model_parser.add_argument(
        '--model_base_name_or_path',
        default="PKU-Alignment/alpaca-7b-reproduced",
        type=str,
        help='the name or path of model to load from',
    )

    # added by us
    model_parser.add_argument(
        '--merged_model_path',
        default='/path/to/model_cache/merged_model', 
        type=str,
        help='the path of the merged model weight to save to',
    )
    model_parser.add_argument(
        '--alpha_helpfulness',
        type=float,
        help='M = M_base + alpha_helpfulness * M_helpfulness + alpha_harmlessness * M_harmlessness (+ alpha_humor * M_humor)',
    )
    model_parser.add_argument(
        '--alpha_harmlessness',
        type=float,
        help='M = M_base + alpha_helpfulness * M_helpfulness + alpha_harmlessness * M_harmlessness (+ alpha_humor * M_humor)',
    )
    model_parser.add_argument(
        '--alpha_humor',
        type=float,
        help='M = M_base + alpha_helpfulness * M_helpfulness + alpha_harmlessness * M_harmlessness (+ alpha_humor * M_humor)',
    )

    model_parser.add_argument(
        '--max_new_tokens',
        type=int,
        default=512,
        help='The maximum sequence length of the generation.',
    )
    model_parser.add_argument(
        '--max_length',
        type=int,
        default=512,
        help='The maximum sequence length of the model.',
    )

    model_parser.add_argument(
        '--normalize_logit',
        type=str2bool,
        default=False,
        help='If True, the temperature argument to ModelArithmetic will be enforced to 1.0, and thus the logit weights of base LLM and ARM sum up to 1; If false, when using ARM decoding, the temperature will be 1/(1+alpha_1+alpha_2).',
    )

    # Dataset
    dataset_parser = parser.add_argument_group('dataset')
    dataset_parser.add_argument(
        '--datasets',
        type=str,
        default="PKU_SafeRLHF_30K"
    )
    dataset_parser.add_argument(
        '--batch_size',
        type=int,
        default=100,
        help='Batch size for evaluation.',
    )

    # Logging
    logging_parser = parser.add_argument_group('logging')
    logging_parser.add_argument(
        '--output_dir',
        type=str,
        default="./results",
        help='Where to store the evaluation output.',
    )
    logging_parser.add_argument(
        '--resume',
        type=str2bool,
        default=False,
    )

    args = parser.parse_args()
    return args


def prompt_template_tinyllama(input_string):
    prompt_split = input_string.split('\n\n')
    messages = []
    for pp in prompt_split:
        if len(pp) == 0:
            continue
        if 'Human:' in pp:
            messages.append({'role': 'user', 'content': (pp.split('Human:')[-1]).strip()})
        if 'Assistant:' in pp:
            messages.append({'role': 'assistant', 'content': (pp.split('Assistant:')[-1]).strip()})

    message_text = ""
    for message in messages[:-1]:
        if message["role"] == "system":
            message_text += "<|system|>\n" + message["content"].strip() + '</s>' + "\n"
        elif message["role"] == "user":
            message_text += "<|user|>\n" + message["content"].strip() + '</s>' + "\n"
        elif message["role"] == "assistant":
            message_text += "<|assistant|>\n" + message["content"].strip() + '</s>' + "\n"
        else:
            raise ValueError("Invalid role: {}".format(message["role"]))
    message_text += "<|assistant|>\n"
    return message_text


def prompt_template_llama(input_string):
    prompt_split = input_string.split('\n\n')
    messages = []
    for pp in prompt_split:
        if len(pp) == 0:
            continue
        if 'Human:' in pp:
            messages.append({'role': 'user', 'content': (pp.split('Human:')[-1]).strip()})
        if 'Assistant:' in pp:
            messages.append({'role': 'assistant', 'content': (pp.split('Assistant:')[-1]).strip()})

    message_text = ""
    for message in messages[:-1]:
        if message["role"] == "user":
            message_text += '<s>' + '[INST] ' + message["content"].strip() + ' [/INST]'
        elif message["role"] == "assistant":
            message_text += ' '  + message["content"].strip() + ' ' + '</s>'
        else:
            raise ValueError("Invalid role: {}".format(message["role"]))
    message_text += ' '
    return message_text


if __name__ == '__main__':

    args = parse_arguments()

    if args.datasets == "PKU_SafeRLHF_30K":
        model_name = f'FedPA_{args.alpha_helpfulness}help_{args.alpha_harmlessness}harm'
    elif args.datasets == "HH_RLHF":
        model_name = f'FedPA_{args.alpha_helpfulness}help_{args.alpha_harmlessness}harm_{args.alpha_humor}humor'
    else:
        raise ValueError(f"Invalid datasets: {args.datasets}")

    ##################### output path #####################
    out_path = Path(os.path.join(args.output_dir, model_name, 'generation') + f".json")
    os.makedirs(os.path.join(args.output_dir, model_name), exist_ok=True)
    print(f'Generation results will be saved in {out_path}')

    ##################### Load dataset #####################
    if args.datasets == "PKU_SafeRLHF_30K":
        data_file_name = "../data/test_prompt_only_PKU_30K.json"
    elif args.datasets == "HH_RLHF":
        data_file_name = "../data/test_prompt_only_HH_RLHF.json"
    else:
        raise ValueError(f"Invalid datasets: {args.datasets}")
        
    with open(data_file_name, 'r') as f:
        data_evaluation = json.load(f)

    ##################### Load models #####################   
    # load the merged model with vLLM
    tensor_parallel_size=2
    print(f"\nLoading merged model on {tensor_parallel_size} GPUs...")
    model = LLM(model=args.merged_model_path, tensor_parallel_size=tensor_parallel_size)
    
    temperature = 0 
    if args.normalize_logit:
        print('\nEnforcing temperature=1.0 in the model_arithmetic generation; The logit weights of base models and potential ARMs are normalized.\n')
        temperature = 1.0
    
    sampling_params = SamplingParams(temperature=temperature, max_tokens=args.max_length)      

    if args.normalize_logit:
        model_name += '_NormalizedLogit'

    print(f'\nModel Name: {model_name}')

    ##################### Generate responses #####################
    output_set = []
    start_index = 0

    start_time_script = time.time()
    for i in tqdm(range(start_index, len(data_evaluation), args.batch_size)):
        prompts = [data_evaluation[j]['prompt'] for j in range(i, min(i + args.batch_size, len(data_evaluation)))]
        
        if args.datasets == "HH_RLHF":
            if 'tinyllama' in args.model_base_name_or_path.lower():
                prompt_inputs = [prompt_template_tinyllama(prompt) for prompt in prompts]
            else:
                prompt_inputs = [prompt_template_llama(prompt) for prompt in prompts]
        else:
            prompt_inputs = [PROMPT_INPUT_ALPACA.format(input=prompt) for prompt in prompts]
        
        start = time.time()

        outputs = model.generate(prompt_inputs,sampling_params)
    
        for j, output in enumerate(outputs):
            response = output.outputs[0].text
            
            elapsed = time.time() -start

            output_set.append({
                "uid": data_evaluation[i + j]['uid'],
                "prompt": prompts[j], 
                "response": response,
                "model": model_name,
                "elapsed":elapsed,
                })

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output_set, f, ensure_ascii=False, indent=4)
    print(f'Generating responses finished!\nSaving to {out_path}\nTime:{(time.time()-start_time_script)/ 3600} hours for {len(output_set) - start_index} outputs')
