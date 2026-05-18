from datasets import load_dataset
from typing import Dict
from functools import partial
import torch
import os


def get_PKU_SafeRLHF(
    dataset_name: str,
    sanity_check: bool = False,
    num_proc=4,
):

    if dataset_name == "PKU_SafeRLHF_30K":
        
        train_dataset_C1 = load_dataset("json", data_files='../data/train_PKU_30K_C1.json', split='train', num_proc=num_proc)
        train_dataset_C2 = load_dataset("json", data_files='../data/train_PKU_30K_C2.json', split='train', num_proc=num_proc)
        
        test_dataset_C1 = load_dataset("json", data_files='../data/dev_PKU_30K_C1.json', split='train', num_proc=num_proc)
        test_dataset_C2 = load_dataset("json", data_files='../data/dev_PKU_30K_C2.json', split='train', num_proc=num_proc)
        
        original_columns = train_dataset_C1.column_names

    else:
        raise ValueError(f"Invalid dataset_name: {dataset_name}")

    if sanity_check:
        train_dataset = train_dataset.select(range(min(len(train_dataset), 1000)))

    ##### chat template #####
    # this is from https://github.com/PKU-Alignment/safe-rlhf/blob/main/safe_rlhf/configs/constants.py
    PROMPT_BEGIN_pku_safe_rlhf: str = 'BEGINNING OF CONVERSATION: '
    PROMPT_USER_pku_safe_rlhf: str = 'USER: {input} '
    PROMPT_ASSISTANT_pku_safe_rlhf: str = 'ASSISTANT:'  # should not have a space at the end

    def format_prompt_pku_safe_rlhf(
        input: str,  
        eos_token: str,
    ) -> str:
        assert isinstance(input, str), f'Unsupported type of `input`: {type(input)}. Expected: str.' 

        if isinstance(input, str):
            input = [input]
        elif not isinstance(input, list):
            raise ValueError(f'Unsupported type of `input`: {type(input)}. Expected: str or list[str].')

        if len(input) % 2 != 1:
            raise ValueError(
                'The length of `input` must be odd, while `input` must end at the user question.',
            )

        buffer = [PROMPT_BEGIN_pku_safe_rlhf]
        for i, line in enumerate(input):
            if i % 2 == 0:
                # User input
                buffer.extend((PROMPT_USER_pku_safe_rlhf.format(input=line), PROMPT_ASSISTANT_pku_safe_rlhf))
            else:
                # Assistant response
                buffer.extend((line, eos_token))

        return ''.join(buffer)
    ##### chat template ends #####
    
    # added by us
    def return_prompt_and_responses(sample, version) -> Dict[str, str]:
        assert version in ["Safe", "Helpful"], "version must be either Safe or Helpful"
        if version == "Safe":
            return {
                "prompt": format_prompt_pku_safe_rlhf(input=sample["prompt"], eos_token='Not used'), 
                "chosen": sample[f"response_{sample['safer_response_id']}"],
                "rejected": sample[f"response_{int(1 - sample['safer_response_id'])}"],
            }
        else:
            return {
                "prompt": format_prompt_pku_safe_rlhf(input=sample["prompt"], eos_token='Not used'),
                "chosen": sample[f"response_{sample['better_response_id']}"],
                "rejected": sample[f"response_{int(1 - sample['better_response_id'])}"],
            }
    
    return_prompt_and_responses_safe = lambda x: return_prompt_and_responses(x, "Safe")
    return_prompt_and_responses_helpful = lambda x: return_prompt_and_responses(x, "Helpful")

    # need to set batched=False because return_prompt_and_responses_with_version can only be applied to a single sample
    if dataset_name == "PKU_SafeRLHF_30K":
        return {'safe': train_dataset_C1.map(
            return_prompt_and_responses_safe,
            batched=False, 
            num_proc=num_proc,
            remove_columns=original_columns,
            keep_in_memory=True,  # keep in memory to speed up the process   
        ), 'helpful': train_dataset_C2.map(
            return_prompt_and_responses_helpful,
            batched=False,
            num_proc=num_proc,
            remove_columns=original_columns,
            keep_in_memory=True,  # keep in memory to speed up the process
        )}, {'safe': test_dataset_C1.map(
            return_prompt_and_responses_safe,
            batched=False, 
            num_proc=num_proc,
            remove_columns=original_columns,
            keep_in_memory=True,  # keep in memory to speed up the process
        ), 'helpful': test_dataset_C2.map(
            return_prompt_and_responses_helpful,
            batched=False,
            num_proc=num_proc,
            remove_columns=original_columns,
            keep_in_memory=True,  # keep in memory to speed up the process
        )}
    else:
        raise ValueError(f"Invalid dataset_name: {dataset_name}")  


def encode_with_messages_format(example, tokenizer, max_seq_length, obj_key, template_type, add_bos=False):
    """
    Here we assume each example has a rejected and chosen field, both of which are a list of messages.
    Each message is a dict with 'role' and 'content' fields.
    We concatenate all messages with the roles as delimiters and tokenize them together.
    We assume only the last message is different, and the prompt is contained in the list of messages.
    """
    # chosen_messages = example["response_0"]
    # rejected_messages = example["response_1"]
    
    chosen_messages = example[f"response_{example[f'{obj_key}_response_id']}"]    # added by us
    rejected_messages = example[f"response_{int(1 - example[f'{obj_key}_response_id'])}"]    # added by us
    
    if len(chosen_messages) == 0:
        raise ValueError("chosen messages field is empty.")
    if len(rejected_messages) == 0:
        raise ValueError("rejected messages field is empty.")

    def _concat_messages(messages):
        return tokenizer.apply_chat_template(messages, tokenize=False)

    def encode_messages(messages):
        example_text = _concat_messages(messages).strip()
        if add_bos:
            example_text = tokenizer.bos_token + example_text
        tokenized_example = tokenizer(example_text, return_tensors="pt", max_length=max_seq_length, truncation=True)
        input_ids = tokenized_example.input_ids
        labels = input_ids.clone()

        # mask the non-assistant part for avoiding loss
        for message_idx, message in enumerate(messages):
            if message["role"] != "assistant":
                if message_idx == 0:
                    message_start_idx = 0
                else:
                    message_start_idx = tokenizer(
                        _concat_messages(messages[:message_idx]),
                        return_tensors="pt",
                        max_length=max_seq_length,
                        truncation=True,
                    ).input_ids.shape[1]
                if message_idx < len(messages) - 1 and messages[message_idx + 1]["role"] == "assistant":
                    # here we also ignore the role of the assistant
                    messages_so_far = _concat_messages(messages[: message_idx + 1])
                    if template_type == 'tinyllama':
                        messages_so_far += "<|assistant|>\n"
                else:
                    messages_so_far = _concat_messages(messages[: message_idx + 1])
                message_end_idx = tokenizer(
                    messages_so_far, return_tensors="pt", max_length=max_seq_length, truncation=True
                ).input_ids.shape[1]
                labels[:, message_start_idx:message_end_idx] = -100

                if message_end_idx >= max_seq_length:
                    break

        attention_mask = torch.ones_like(input_ids)
        return {
            "input_ids": input_ids.flatten(),
            "labels": labels.flatten(),
            "attention_mask": attention_mask.flatten(),
        }

    chosen_encoded = encode_messages(chosen_messages)
    rejected_encoded = encode_messages(rejected_messages)
    # labels are useful for working out where the loss is valid.
    
    return {
        "chosen_input_ids": chosen_encoded["input_ids"],
        "chosen_labels": chosen_encoded["labels"],
        "chosen_attention_mask": chosen_encoded["attention_mask"],
        "rejected_input_ids": rejected_encoded["input_ids"],
        "rejected_labels": rejected_encoded["labels"],
        "rejected_attention_mask": rejected_encoded["attention_mask"],
        # "labels": {obj: example[f'{obj}_response_id'] for obj in obj_key}
    }


def get_HH_RLHF(
    tokenizer,
    template_type,
    num_proc=4,
):
    
    train_dataset_C1 = load_dataset("json", data_files='../data/train_HH_RLHF_C1.json', split='train', num_proc=num_proc)
    test_dataset_C1 = load_dataset("json", data_files='../data/dev_HH_RLHF_C1.json', split='train', num_proc=num_proc)
    
    train_dataset_C2 = load_dataset("json", data_files='../data/train_HH_RLHF_C2.json', split='train', num_proc=num_proc)
    test_dataset_C2 = load_dataset("json", data_files='../data/dev_HH_RLHF_C2.json', split='train', num_proc=num_proc)
    
    train_dataset_C3 = load_dataset("json", data_files='../data/train_HH_RLHF_C3.json', split='train', num_proc=num_proc)
    test_dataset_C3 = load_dataset("json", data_files='../data/dev_HH_RLHF_C3.json', split='train', num_proc=num_proc)

    def map_dataset(given_dataset, obj_key):
        
        encode_function = partial(
                encode_with_messages_format,
                tokenizer=tokenizer,
                max_seq_length=1024,
                obj_key=obj_key,
                template_type=template_type
        )
        
        lm_datasets = given_dataset.map(
            encode_function,
            batched=False,
            num_proc=num_proc,
            keep_in_memory=True,  # keep in memory to speed up the process
            remove_columns=[
                name
                for name in given_dataset.column_names
                if name
                not in [
                    "chosen_input_ids",
                    "chosen_labels",
                    "chosen_attention_mask",
                    "rejected_input_ids",
                    "rejected_labels",
                    "rejected_attention_mask",
                    # "labels",
                ]
            ],
            desc="Tokenizing and reformatting instruction data",
        )
        lm_datasets.set_format(type="pt")
        # our thresholding mighta meant some examples have no labels, remove.
        lm_datasets = lm_datasets.filter(lambda example: (example["chosen_labels"] != -100).any())
        lm_datasets = lm_datasets.filter(lambda example: (example["rejected_labels"] != -100).any())
        return lm_datasets

    # need to set batched=False because return_prompt_and_responses_with_version can only be applied to a single sample
    return {'help': map_dataset(train_dataset_C1, 'help'),
            'harm': map_dataset(train_dataset_C2, 'harm'),
            'humor': map_dataset(train_dataset_C3, 'humor'),
            }, {'help': map_dataset(test_dataset_C1, 'help'),
            'harm': map_dataset(test_dataset_C2, 'harm'),
            'humor': map_dataset(test_dataset_C3, 'humor'),}


######################################################################
# VLFeedback dataset for Qwen3-VL
######################################################################

def return_vl_prompt_and_responses(example, obj_key, data_dir):
    """
    Select chosen/rejected based on per-dimension preference ID.
    Stores image_path (str) instead of PIL Image to keep dataset memory-light.
    Images are loaded on-the-fly in the data collator.
    """
    chosen_response = example[f"response_{example[f'{obj_key}_response_id']}"]
    rejected_response = example[f"response_{int(1 - example[f'{obj_key}_response_id'])}"]

    image_path = os.path.join(data_dir, example["image"])

    return {
        "prompt": example["prompt"],
        "chosen": chosen_response,
        "rejected": rejected_response,
        "image_path": image_path,
    }


def get_VLFeedback(processor, num_proc=4):
    """
    Load VLFeedback dataset for DPO training with Qwen3-VL.

    3 objectives/clients: help (helpfulness), faith (visual faithfulness), ethic (ethical considerations).
    Returns: (train_datasets, eval_datasets) each dict of {obj_name: Dataset}

    Dataset only stores text fields + image_path (str). Images are loaded
    on-the-fly in the data collator to keep memory usage low.
    """
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')

    train_dataset_C1 = load_dataset("json", data_files=os.path.join(data_dir, 'train_VLFeedback_C1.json'), split='train', num_proc=num_proc)
    train_dataset_C2 = load_dataset("json", data_files=os.path.join(data_dir, 'train_VLFeedback_C2.json'), split='train', num_proc=num_proc)
    train_dataset_C3 = load_dataset("json", data_files=os.path.join(data_dir, 'train_VLFeedback_C3.json'), split='train', num_proc=num_proc)

    test_dataset_C1 = load_dataset("json", data_files=os.path.join(data_dir, 'dev_VLFeedback_C1.json'), split='train', num_proc=num_proc)
    test_dataset_C2 = load_dataset("json", data_files=os.path.join(data_dir, 'dev_VLFeedback_C2.json'), split='train', num_proc=num_proc)
    test_dataset_C3 = load_dataset("json", data_files=os.path.join(data_dir, 'dev_VLFeedback_C3.json'), split='train', num_proc=num_proc)

    output_cols = {"prompt", "chosen", "rejected", "image_path"}

    def map_dataset(given_dataset, obj_key):
        map_fn = partial(return_vl_prompt_and_responses, obj_key=obj_key, data_dir=data_dir)
        original_columns = given_dataset.column_names
        mapped = given_dataset.map(
            map_fn,
            batched=False,
            num_proc=num_proc,
            remove_columns=[c for c in original_columns if c not in output_cols],
            desc=f"Processing VLFeedback ({obj_key})",
        )
        return mapped

    return {
        'help': map_dataset(train_dataset_C1, 'help'),
        'faith': map_dataset(train_dataset_C2, 'faith'),
        'ethic': map_dataset(train_dataset_C3, 'ethic'),
    }, {
        'help': map_dataset(test_dataset_C1, 'help'),
        'faith': map_dataset(test_dataset_C2, 'faith'),
        'ethic': map_dataset(test_dataset_C3, 'ethic'),
    }