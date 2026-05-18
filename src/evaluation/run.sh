export CUDA_VISIBLE_DEVICES=0,1

base_model_dir="/path/to/model_cache/model--PKU-Alignment--alpaca-7b-reproduced"
merged_model_dir="/path/to/model_cache/merged_model/"

datasets=PKU_SafeRLHF_30K
exp_dir="../training/ckpt/exp_dir/final_checkpoint"
output_dir="./results"
cache_dir="./cache_ours"

for alpha in $(seq 0 0.1 1); do
    alpha_safe=$(printf "%.1f" $(bc <<< "1 - $alpha"))
    
    echo "Processing alpha=$alpha, alpha_safe=$alpha_safe"
    
    # 1. merge model
    python merge_model.py \
        --cache_dir "$cache_dir" \
        --alpha_helpfulness "$alpha" \
        --alpha_harmlessness "$alpha_safe" \
        --model_base_name_or_path "$base_model_dir"\
        --model_fedpa_both_name_or_path "$exp_dir" \
        --save_path "$merged_model_dir"
    
    # 2. generate outputs
    python generate_outputs.py \
        --output_dir "$output_dir" \
        --alpha_helpfulness "$alpha" \
        --alpha_harmlessness "$alpha_safe" \
        --merged_model_path "$merged_model_dir" \
        --datasets "$datasets" \
        --batch_size 100
    
    # 3. compute reward
    result_path="${output_dir}/FedPA_${alpha}help_${alpha_safe}harm"
    python compute_reward.py \
        --path "$result_path"
    
    # 4. clean up
    echo "Cleaning up $merged_model_dir"
    rm -rf "${merged_model_dir}"*
done
