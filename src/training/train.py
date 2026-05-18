'''
adapted from trl/examples/research_projects/stack_llama_2/scripts/dpo_llama2.py

Due to data processing steps (data format and model template), this script only supports training LLaMA-1 on the HH-RLHF dataset and training Alpaca on the PKU-SafeRLHF dataset.
For other dataset and models, we need to modify the data processing steps.
'''
import os
import sys
import gc
import logging
from dataclasses import dataclass, field
from typing import Optional
import torch
from accelerate import Accelerator
from peft import PCLoraConfig, LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import DPOTrainer, DPOConfig
from trl.commands.cli_utils import TrlParser
from scipy.stats import dirichlet
import copy
import random
from load_data import get_PKU_SafeRLHF, get_HH_RLHF, get_VLFeedback
from HH_RLHF_DPOTrainer import HH_RLHF_DPOTrainer
from VLFeedback_DPOTrainer import VLFeedback_DPOTrainer
import wandb
wandb.init(mode="disabled")


@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.

    NOTE: other training arguments, such as learning rate and beta, should be set in the command line.
    They are included in DPOConfig, not here. ScriptArguments below are arguments that are not included in DPOConfig.
    """
    # used by Ours
    algorithm: Optional[str] = field(default="local", metadata={"help": "algorithm to use [local, fedavg, fedpa]"})
    model_name_or_path: Optional[str] = field(
        default="PKU-Alignment/alpaca-7b-reproduced",
        metadata={"help": "the location of the to-be-finetuned model name or path"},
    )

    # dataset
    preference_dataset: Optional[str] = field(default="PKU_SafeRLHF_30K", metadata={"help": ""})

    # training
    optimizer_type: Optional[str] = field(default="paged_adamw_32bit", metadata={"help": "the optimizer type"})

    # model
    lora_r: Optional[int] = field(default=8, metadata={"help": "the lora r parameter"})
    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    
    load_in_4bit: Optional[bool] = field(default=True, metadata={"help": "whether to load the model in 4bit"})
    model_dtype: Optional[str] = field(default="float16", metadata={"help": "model_dtype[float16, bfloat16, float] for loading."})

    # others
    sanity_check: Optional[bool] = field(default=False, metadata={"help": "only train on 1000 samples"})
    ignore_bias_buffers: Optional[bool] = field(
        default=False,
        metadata={
            "help": "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        },
    )
    
    pref_sample_p: Optional[float] = field(default=1.0, metadata={"help": "Dirichlet distribution parameter for sampling preference vector"})
    communication_rounds: Optional[int] = field(default=100, metadata={"help": "the number of communication rounds for federated learning"})


if __name__ == "__main__":

    parser = TrlParser((ScriptArguments, DPOConfig))
    script_args, training_args = parser.parse_args_and_config()

    # Setup per-experiment log file in output_dir (main process only)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    os.makedirs(training_args.output_dir, exist_ok=True)
    log_path = os.path.join(training_args.output_dir, "training.log")
    if local_rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(log_path, mode="a"),
                logging.StreamHandler(sys.stdout),
            ],
        )
    else:
        logging.basicConfig(level=logging.WARNING)
    logger = logging.getLogger(__name__)
    logger.info(f"Logging to {log_path}")
    logger.info(f"Script args: {script_args}")
    logger.info(f"Training args: {training_args}")

    logger.info(f'Preference dataset: {script_args.preference_dataset}')

    training_args.gradient_checkpointing_kwargs={"use_reentrant": False} # this is necessary due to https://github.com/huggingface/trl/issues/480

    set_seed(training_args.seed)
    
    # 2-1. define model type and load tokenizer
    torch_dtype = torch.float
    if script_args.model_dtype == "float16":
        torch_dtype = torch.float16
    elif script_args.model_dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    logger.info(f'model_name_or_path: {script_args.model_name_or_path}')

    is_vlm = script_args.preference_dataset == "VLFeedback"

    # For text-only DPO (PKU_SafeRLHF_30K / HH_RLHF), gradient checkpointing can trigger
    # PyTorch checkpoint recompute issues. We disable checkpointing for stability.
    if not is_vlm:
        training_args.gradient_checkpointing = False

    if is_vlm:
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
        processor = AutoProcessor.from_pretrained(
            script_args.model_name_or_path,
            min_pixels=3136,      # 56*56
            max_pixels=1003520,   # 28*28*1280, ~980 image tokens per image
        )
        tokenizer = processor.tokenizer
        tokenizer.pad_token = tokenizer.eos_token
    else:
        processor = None
        tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)
        tokenizer.pad_token = tokenizer.eos_token

    # 1. Load the preference dataset
    if script_args.preference_dataset == "PKU_SafeRLHF_30K":
        train_datasets, eval_datasets = get_PKU_SafeRLHF(dataset_name=script_args.preference_dataset, sanity_check=script_args.sanity_check)
    elif script_args.preference_dataset == "HH_RLHF":
        train_datasets, eval_datasets = get_HH_RLHF(tokenizer=tokenizer, template_type='tinyllama' if 'tinyllama' in script_args.model_name_or_path.lower() else 'llama')
    elif script_args.preference_dataset == "VLFeedback":
        train_datasets, eval_datasets = get_VLFeedback(processor=processor)
    else:
        raise ValueError(f"Invalid preference dataset: {script_args.preference_dataset}")
    # print(train_datasets)
    
    for obj in train_datasets.keys():
        logger.info(f'Before filtering. {obj} train data size: {train_datasets[obj].num_rows}, Test data size: {eval_datasets[obj].num_rows}')
        
    if script_args.preference_dataset != "HH_RLHF" and script_args.preference_dataset != "VLFeedback":
        for obj in train_datasets.keys():
            train_datasets[obj] = train_datasets[obj].filter(
                lambda x: len(x["prompt"]) + len(x["chosen"]) <= training_args.max_length
                and len(x["prompt"]) + len(x["rejected"]) <= training_args.max_length
            )
            eval_datasets[obj] = eval_datasets[obj].filter(
                lambda x: len(x["prompt"]) + len(x["chosen"]) <= training_args.max_length
                and len(x["prompt"]) + len(x["rejected"]) <= training_args.max_length
            )

        for obj in train_datasets.keys():
            logger.info(f'After filtering. {obj} train data size: {train_datasets[obj].num_rows}, Test data size: {eval_datasets[obj].num_rows}')
    
    # Helper: load model (VLM or CausalLM)
    def load_base_model():
        if is_vlm:
            m = Qwen3VLForConditionalGeneration.from_pretrained(
                script_args.model_name_or_path,
                low_cpu_mem_usage=True,
                torch_dtype=torch_dtype,
                load_in_4bit=script_args.load_in_4bit,
                device_map={"": Accelerator().local_process_index},
            )
        else:
            m = AutoModelForCausalLM.from_pretrained(
                script_args.model_name_or_path,
                low_cpu_mem_usage=True,
                torch_dtype=torch_dtype,
                load_in_4bit=script_args.load_in_4bit,
                device_map={"": Accelerator().local_process_index},
            )
        m.config.use_cache = False
        if script_args.ignore_bias_buffers:
            m._ddp_params_and_buffers_to_ignore = [
                name for name, buffer in m.named_buffers() if buffer.dtype == torch.bool
            ]
        return m

    # Helper: create the appropriate DPO trainer
    def create_trainer(model, train_dataset, eval_dataset, peft_config, ref_model=None):
        if script_args.preference_dataset == "VLFeedback":
            return VLFeedback_DPOTrainer(
                processor=processor,
                model=model,
                ref_model=ref_model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=processor,
                peft_config=peft_config,
            )
        elif script_args.preference_dataset == "HH_RLHF":
            return HH_RLHF_DPOTrainer(
                model=model,
                ref_model=ref_model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=tokenizer,
                peft_config=peft_config,
            )
        else:
            return DPOTrainer(
                model=model,
                ref_model=ref_model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=tokenizer,
                peft_config=peft_config,
            )

    # 3. initialize training arguments (done in DPOconfig) and peft config

    if script_args.algorithm == "fedpa":
        peft_config = PCLoraConfig(
            r=script_args.lora_r,
            obj_num=len(train_datasets.keys()),
            lora_alpha=script_args.lora_alpha,
            lora_dropout=script_args.lora_dropout,
            target_modules=[
                "q_proj",
                "v_proj",
                "k_proj",
            ],
            bias="none",
            task_type="CAUSAL_LM",
        )
    else:
        peft_config = LoraConfig(
            r=script_args.lora_r,
            lora_alpha=script_args.lora_alpha,
            lora_dropout=script_args.lora_dropout,
            target_modules=[
                "q_proj",
                "v_proj",
                "k_proj",
            ],
            bias="none",
            task_type="CAUSAL_LM",
        )
        
    if script_args.algorithm == "local":
        logger.info('********************************')
        logger.info('Using Local Finetuning')
        logger.info('********************************\n')
        for obj in train_datasets.keys():
            logger.info(f'Local finetuning on {obj} dataset')

            # local dataset
            train_dataset = train_datasets[obj]
            eval_dataset = eval_datasets[obj]

            # 2-2. load a pretrained model
            model = load_base_model()

            # 4. initialize the DPO trainer
            trainer = create_trainer(model, train_dataset, eval_dataset, peft_config)

            # 5. train
            trainer.train()
            trainer.save_model(training_args.output_dir)

            # 6. save
            output_dir = os.path.join(training_args.output_dir, f"final_checkpoint_{obj}")
            trainer.model.save_pretrained(output_dir)
    
    elif script_args.algorithm == "fedavg":
        logger.info('********************************')
        logger.info(f'Using {script_args.algorithm} Finetuning')
        logger.info('********************************\n')

        # 2.2 load a pretrained model
        model = load_base_model()

        preference = torch.tensor([1 / len(train_datasets.keys())] * len(train_datasets.keys()))     # for different preference combinations
        final_trainable_params = {}  # trainable parameters
        for t in range(script_args.communication_rounds):
            logger.info(f'Communication round {t + 1}/{script_args.communication_rounds}')
            global_trainable_params = {}   # trainable parameters

            idx = 0   # used for preference vector
            for obj in train_datasets.keys():
                # objs: safe, helpful
                logger.info(f'Local finetuning on {obj} dataset')

                train_dataset = train_datasets[obj]
                eval_dataset = eval_datasets[obj]

                # make a copy of the model for local finetuning
                local_model = copy.deepcopy(model)
                local_model.config.use_cache = False

                # 4. initialize the DPO trainer
                trainer = create_trainer(local_model, train_dataset, eval_dataset, peft_config)

                if t > 0:
                    trainer.model.load_state_dict(final_trainable_params, strict=False)    # load the model weight from the previous round
                
                # 5. train
                trainer.train()
                
                # 6. update the global trainable parameters of the model
                if global_trainable_params == {}:
                    global_trainable_params = {name: param.data.detach().cpu().clone() * preference[idx] for name, param in trainer.model.named_parameters() if param.requires_grad}
                else:
                    for name, param in trainer.model.named_parameters():
                        if param.requires_grad:
                                global_trainable_params[name].data += param.data.detach().cpu().clone() * preference[idx]
                idx += 1    # next obj / client

                del trainer, local_model
                gc.collect()
                torch.cuda.empty_cache()

            # 7. update the final trainable parameters
            final_trainable_params = {name: param.data.clone() for name, param in global_trainable_params.items()}
        
        # 8. save the server model
        output_dir = os.path.join(training_args.output_dir, "final_checkpoint")
        model = get_peft_model(model, peft_config)
        model.load_state_dict(final_trainable_params, strict=False)  # load the final trainable parameters
        model.save_pretrained(output_dir)
    
    elif script_args.algorithm == "fedpa":
        logger.info('********************************')
        logger.info(f'Using {script_args.algorithm} Finetuning')
        logger.info('********************************\n')
        random_numbers = random.sample(range(0, 10000 + 1), script_args.communication_rounds)

        # 2.2 load a pretrained model
        model = load_base_model()

        final_trainable_params = {}  # final trainable parameters
        for t in range(script_args.communication_rounds):
            logger.info(f'Communication round {t + 1}/{script_args.communication_rounds}')
            training_args.seed = random_numbers[t]   # set a different seed for each communication round for sampling preference vector and training data
            
            # sample a preference vector
            preference = torch.tensor(dirichlet.rvs([script_args.pref_sample_p]*len(train_datasets.keys())))[0] # sample a preference vector from Dirichlet distribution
            logger.info(f'Sampled preference vector: {preference}')
            
            all_trainable_params = {}   # all trainable parameters
            # Alternating training
            para_idx = 0
            for trainable_para in ['pclora_B', 'pclora_W', 'pclora_A']:
                global_trainable_params = {}   # trainable parameters
                
                # for each client
                idx = 0   # used for preference vector
                for obj in train_datasets.keys():
                    logger.info(f'Local finetuning parameters {trainable_para} on {obj} dataset')

                    train_dataset = train_datasets[obj]
                    eval_dataset = eval_datasets[obj]
                                
                    # make a copy of the model for local finetuning
                    local_model = copy.deepcopy(model)
                    local_model.config.use_cache = False
                
                    # 4. initialize the DPO trainer
                    trainer = create_trainer(local_model, train_dataset, eval_dataset, peft_config)

                    # freeze other parameters
                    for name, param in trainer.model.named_parameters():
                        if trainable_para in name:
                            param.requires_grad = True
                        else:
                            param.requires_grad = False
                    
                    if t > 0:
                        trainer.model.load_state_dict(final_trainable_params, strict=False)    # load the model weight from the previous round
                    
                    if para_idx > 0:
                        trainer.model.load_state_dict(all_trainable_params, strict=False)    # load the model weight from the previous fine-tuning procedure

                    # set the preference vector
                    for n, p in trainer.model.named_parameters():
                        if 'pref_vec' in n:
                            p.data = preference.to(p.device)
                            p.requires_grad = False
                    
                    # 5. train
                    trainer.train()
                
                    # 6. update the global trainable parameters
                    if global_trainable_params == {}:
                        global_trainable_params = {name: param.data.detach().cpu().clone() * preference[idx].cpu() for name, param in trainer.model.named_parameters() if param.requires_grad}
                    else:
                        for name, param in trainer.model.named_parameters():
                            if param.requires_grad:
                                    global_trainable_params[name].data += param.data.detach().cpu().clone() * preference[idx].cpu()
                    idx += 1    # next obj / client

                    del trainer, local_model
                    gc.collect()
                    torch.cuda.empty_cache()

                # 7. update all trainable parameters
                all_trainable_params.update(global_trainable_params)  # extend the all trainable parameters

                para_idx += 1    # next trainable parameter

            # 8. update the final trainable parameters (keep on CPU to save GPU memory)
            final_trainable_params = {name: param.data.clone() for name, param in all_trainable_params.items()}
            del all_trainable_params
            gc.collect()
            torch.cuda.empty_cache()
            
            # 9. save intermediate server model every 50 communication rounds
            if (t + 1) % 50 == 0:
                logger.info(f'Saving intermediate server model at communication round {t + 1}')
                intermediate_output_dir = os.path.join(training_args.output_dir, f"intermediate_checkpoint_round_{t + 1}")
                os.makedirs(intermediate_output_dir, exist_ok=True)
                
                # Create a temporary model by deep copying the base model
                temp_model = copy.deepcopy(model)
                temp_model.config.use_cache = False
                temp_model = get_peft_model(temp_model, peft_config)
                temp_model.load_state_dict(final_trainable_params, strict=False)
                temp_model.save_pretrained(intermediate_output_dir)
                
                # Clean up temporary model to save memory
                del temp_model
                torch.cuda.empty_cache()
                logger.info(f'Intermediate model saved to {intermediate_output_dir}')
        
        # 8. save the server model
        output_dir = os.path.join(training_args.output_dir, "final_checkpoint")
        model = get_peft_model(model, peft_config)
        model.load_state_dict(final_trainable_params, strict=False)  # load the final trainable parameters
        model.save_pretrained(output_dir)
    
    else:
        raise ValueError(f"Invalid algorithm: {script_args.algorithm}. Must be either 'local', 'fedavg', or 'fedpa'.")
