cuda=0,1,2,3,4,5,6,7
exp_name=fedpa_exp

lora_r=8
lora_alpha=16

pref_sample_p=0.5

epoch=1
steps=5   # If set to a positive number, the total number of training steps to perform. Overrides `epochs`. [-1, 5]
beta=5e-1
learning_rate=5e-4
bs=32
per_device_train_batch_size=2
communication_rounds=100
algorithm=fedpa
preference_dataset=PKU_SafeRLHF_30K 

model_name_or_path=/path/to/model_cache/model--PKU-Alignment--alpaca-7b-reproduced

###### the following is automatically set
num_GPU=$(echo $cuda | awk -F, '{print NF}')
gradient_accumulation_steps=$(($bs/$num_GPU/$per_device_train_batch_size))

output_dir=./ckpt/exp_${preference_dataset}_${algorithm}_I_${steps}_T_${communication_rounds}_lr_${learning_rate}_beta_${beta}
if [ -d "${output_dir}" ]; then
    echo -e "\n\n"
    echo "Error: Directory "${output_dir}" already exists. Please delete it or choose a new output_dir." >&2
    exit 1
fi
echo "Output dir: $output_dir"

# cd /path/code/training
accelerate launch --gpu_ids $cuda --main_process_port 29500 --num_processes $num_GPU train.py \
    --algorithm=$algorithm \
    --preference_dataset=$preference_dataset \
    --pref_sample_p=$pref_sample_p \
    --lora_r=$lora_r \
    --lora_alpha=$lora_alpha \
    --model_name_or_path=$model_name_or_path \
    --beta=$beta \
    --learning_rate=$learning_rate \
    --num_train_epochs=$epoch \
    --max_steps=$steps \
    --communication_rounds=$communication_rounds \
    --output_dir=$output_dir \
    --run_name=$exp_name \
    --per_device_train_batch_size=$per_device_train_batch_size \
    --gradient_accumulation_steps=$gradient_accumulation_steps \
    --per_device_eval_batch_size=2 \
    --logging_steps=10 \
    --eval_strategy="steps" \
    --eval_steps=20 \
    --save_strategy="steps" \
    --save_steps=1000 \
    --lr_scheduler_type="cosine" \
    --warmup_steps=20 \
    --weight_decay=0.05 \
    --gradient_checkpointing=True \
    --bf16=True \
    --max_prompt_length=512 \
    --max_length=1024 \
    --report_to="wandb" \
    --remove_unused_columns=False  \

echo "Finished training $output_dir"