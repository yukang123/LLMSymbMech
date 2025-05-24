#!/bin/bash
#SBATCH --job-name=llm_abstractor         # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=150G         # total memory
#SBATCH --gres=gpu:2           # number of gpus per node
#SBATCH --constraint=gpu80       # specify the gpu type (gpu80: A100)
#SBATCH --time=9:59:00          # total run time limit (HH:MM:SS)
#SBATCH --output=slurm_logs/%j.out

source activate LLMSymbMech
cd tasks/identity_rules

## 1. eval generation
python eval.py --rule ABA --model_type Llama-3.1-70B --prompt_num 500  --in_context_example_num 2 --sample_size 4 --acc_threshold 0.9 --load_generation_config --sample_remark _greedy
python eval.py --rule ABB --model_type Llama-3.1-70B --prompt_num 500  --in_context_example_num 2 --sample_size 4 --acc_threshold 0.9 --load_generation_config --sample_remark _greedy  


## 2. causal mediation analyses (CMA) 
base_rule=ABA
min_valid_sample_num=20
threshold=0.9
seed=0 ## try 5 different seeds to get the average results over 100 context pairs

## 2 in-context examples (2-shot) for Llama-3.1-70B in the main text
## CMA on different layers x different token positions
## abstract context
python cma.py --activation_name attn_out --patch_mlp_out --context_type abstract --in_context_example_num 2 --base_rule $base_rule --model_type Llama-3.1-70B --seed $seed --min_valid_sample_num $min_valid_sample_num --low_prob_threshold $threshold --do_shuffle --ungroup_grouped_query_attention --prompt_num 1000  --eval_metric gen_acc --load_generation_config --sample_size 4 --n_devices 2 --device_map cpu --device cuda 
## token context
python cma.py --activation_name attn_out --patch_mlp_out --context_type token --in_context_example_num 2 --base_rule $base_rule --model_type Llama-3.1-70B --seed $seed --min_valid_sample_num $min_valid_sample_num --low_prob_threshold $threshold --do_shuffle --ungroup_grouped_query_attention --prompt_num 1000  --eval_metric gen_acc --load_generation_config --sample_size 4 --n_devices 2 --device_map cpu --device cuda 
# -m debugpy --listen 0.0.0.0:$PORT --wait-for-client 

## CMA on different heads
## abstract context
# symbol abstraction heads
python cma.py --activation_name z --context_type abstract --token_pos_list 4 10 --in_context_example_num 10 --base_rule $base_rule --model_type Llama-3.1-70B --seed $seed --min_valid_sample_num $min_valid_sample_num --low_prob_threshold $threshold --do_shuffle --ungroup_grouped_query_attention --prompt_num 1000  --eval_metric gen_acc --load_generation_config --sample_size 4 --n_devices 2 --device_map cpu --device cuda 
# symbolic induction heads
python cma.py --activation_name z --context_type abstract --token_pos_list -1 --in_context_example_num 10 --base_rule $base_rule --model_type Llama-3.1-70B --seed $seed --min_valid_sample_num $min_valid_sample_num --low_prob_threshold $threshold --do_shuffle --ungroup_grouped_query_attention --prompt_num 1000  --eval_metric gen_acc --load_generation_config --sample_size 4 --n_devices 2 --device_map cpu --device cuda 
# retrieval heads
python cma.py --activation_name z --context_type token --token_pos_list -1 --in_context_example_num 10 --base_rule $base_rule --model_type Llama-3.1-70B --seed $seed --min_valid_sample_num $min_valid_sample_num --low_prob_threshold $threshold --do_shuffle --ungroup_grouped_query_attention --prompt_num 1000  --eval_metric gen_acc --load_generation_config --sample_size 4 --n_devices 2 --device_map cpu --device cuda 


## While for systematic analyses of 13 models from 4 families, we do experiments on 10-shot examples (GPT-2, Gemma-2, Qwen2.5, Llama-3.1)
model_type=Llama-3.1-70B
# model_family_dict = {
#     "gpt2": ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"],
#     "Llama-3": ["Llama-3.1-70B", "Llama-3.1-8B"],
#     "Qwen2.5": ["Qwen2.5-7B", "Qwen2.5-14B", "Qwen2.5-32B", "Qwen2.5-72B"],
#     "gemma-2": ["gemma-2-2b", "gemma-2-9b", "gemma-2-27b"], 
# }
n_devices=2 ## for Llama-3.1-70B and Qwen2.5-72B, we use 2 devices to avoid OOM while for other models, we use 1 device
# symbol abstraction heads
python cma.py --activation_name z --context_type abstract --token_pos_list 4 10 16 22 28 34 40 46 52 58 --in_context_example_num 10 --base_rule $base_rule --model_type $model_type --n_devices $n_devices --seed $seed --min_valid_sample_num $min_valid_sample_num --low_prob_threshold $threshold --do_shuffle --ungroup_grouped_query_attention --prompt_num 1000  --eval_metric gen_acc --load_generation_config --sample_size 4  
# symbolic induction heads
python cma.py --activation_name z --context_type abstract --token_pos_list -1 --in_context_example_num 10 --base_rule $base_rule --model_type $model_type --n_devices $n_devices --seed $seed --min_valid_sample_num $min_valid_sample_num --low_prob_threshold $threshold --do_shuffle --ungroup_grouped_query_attention --prompt_num 1000  --eval_metric gen_acc --load_generation_config --sample_size 4 
# retrieval heads
python cma.py --activation_name z --context_type token --token_pos_list -1 --in_context_example_num 10 --base_rule $base_rule --model_type $model_type --n_devices $n_devices --seed $seed --min_valid_sample_num $min_valid_sample_num --low_prob_threshold $threshold --do_shuffle --ungroup_grouped_query_attention --prompt_num 1000  --eval_metric gen_acc --load_generation_config --sample_size 4 


## 3. Representation Similarity Analysis (RSA)
prompt_num=40
## Symbol abstraction head 
python rsa.py --do_shuffle --seed 1234 --only_for_significant_heads --head_type symbol_abstraction_head --cmp_with_abstract --sel_pos_list 4 --in_context_example_num 2 --act_list z --add_swap_1_2_question --prompt_num $prompt_num --plot_hand_code --plot_rsa --transpose --low_sim 0 --token_set_file datasets/llama31_70B_correct_common_tokens_0.9_1378.txt --n_devices 2 --model_type Llama-3.1-70B # --save_similarity 
python rsa.py --do_shuffle --seed 1234 --only_for_significant_heads --head_type symbol_abstraction_head --cmp_with_abstract --sel_pos_list 10 --in_context_example_num 2 --act_list z --add_swap_1_2_question --prompt_num $prompt_num --plot_hand_code --plot_rsa  --transpose --low_sim 0 --token_set_file datasets/llama31_70B_correct_common_tokens_0.9_1378.txt --n_devices 2 --model_type Llama-3.1-70B 
python rsa.py --do_shuffle --seed 1234 --only_for_significant_heads --head_type symbol_abstraction_head --cmp_with_token_id --sel_pos_list 4 --in_context_example_num 2 --act_list z --add_swap_1_2_question --prompt_num $prompt_num --plot_hand_code --plot_rsa  --transpose --low_sim 0 --token_set_file datasets/llama31_70B_correct_common_tokens_0.9_1378.txt --n_devices 2 --model_type Llama-3.1-70B 
python rsa.py --do_shuffle --seed 1234 --only_for_significant_heads --head_type symbol_abstraction_head --cmp_with_token_id --sel_pos_list 10 --in_context_example_num 2 --act_list z --add_swap_1_2_question --prompt_num $prompt_num --plot_hand_code --plot_rsa  --transpose --low_sim 0 --token_set_file datasets/llama31_70B_correct_common_tokens_0.9_1378.txt --n_devices 2 --model_type Llama-3.1-70B 

## Symbolic induction head
python rsa.py --do_shuffle --seed 4567 --only_for_significant_heads --head_type symbolic_induction_head --cmp_with_abstract --sel_pos_list -1 --in_context_example_num 2 --act_list z --add_swap_1_2_question --prompt_num $prompt_num --plot_hand_code --plot_rsa  --transpose --low_sim 0 --token_set_file datasets/llama31_70B_correct_common_tokens_0.9_1378.txt --n_devices 2 --model_type Llama-3.1-70B 
python rsa.py --do_shuffle --seed 4567 --only_for_significant_heads --head_type symbolic_induction_head --cmp_with_token_id --sel_pos_list -1 --in_context_example_num 2 --act_list z --add_swap_1_2_question --prompt_num $prompt_num --plot_hand_code --plot_rsa  --transpose --low_sim 0 --token_set_file datasets/llama31_70B_correct_common_tokens_0.9_1378.txt --n_devices 2 --model_type Llama-3.1-70B 

## Retrieval head
python rsa.py --do_shuffle --seed 7890 --only_for_significant_heads --head_type retrieval_head --cmp_with_abstract --sel_pos_list -1 --in_context_example_num 2 --act_list z --add_swap_1_2_question --prompt_num $prompt_num --plot_hand_code --plot_rsa  --transpose --low_sim 0 --token_set_file datasets/llama31_70B_correct_common_tokens_0.9_1378.txt --n_devices 2 --model_type Llama-3.1-70B 
python rsa.py --do_shuffle --seed 7890 --only_for_significant_heads --head_type retrieval_head --cmp_with_token_id --sel_pos_list -1 --in_context_example_num 2 --act_list z --add_swap_1_2_question --prompt_num $prompt_num --plot_hand_code --plot_rsa  --transpose --low_sim 0 --token_set_file datasets/llama31_70B_correct_common_tokens_0.9_1378.txt --n_devices 2 --model_type Llama-3.1-70B 


## 3. Ablation Studies
rule=ABA # ABA or ABB
min_valid_sample_num=40
seed=1000 ## please specify the seed which is different form the seeds used in CMA
random_times=10
### symbol abstraction head
python ablation.py --model_type Llama-3.1-70B --head_type symbol_abstraction_head --step_size 2 --adaptive_step_size --rule $rule --prompt_file_for_causal_scores_exp "datasets/cma_scores/llama31_70B/symbol_abstraction_head/base_${rule}_prompt_for_cma_100.txt" --token_pos_list 4 10 --activation_name z --min_valid_sample_num $min_valid_sample_num --load_generation_config  --do_shuffle --seed $seed --prompt_num 1000 --low_prob_threshold 0.9 --n_devices 2 --token_set_file datasets/llama31_70B_correct_common_tokens_0.9_1378.txt  --in_context_example_num 2 --eval_metric gen_acc
## control condition
python ablation.py --model_type Llama-3.1-70B --head_type symbol_abstraction_head --step_size 2 --control --adaptive_step_size --rule $rule --prompt_file_for_causal_scores_exp "datasets/cma_scores/llama31_70B/symbol_abstraction_head/base_${rule}_prompt_for_cma_100.txt" --token_pos_list 4 10 --activation_name z --min_valid_sample_num $min_valid_sample_num  --load_generation_config  --do_shuffle --seed $seed --prompt_num 1000 --low_prob_threshold 0.9 --n_devices 2 --token_set_file datasets/llama31_70B_correct_common_tokens_0.9_1378.txt  --in_context_example_num 2 --eval_metric gen_acc
## random control condition
python ablation.py --model_type Llama-3.1-70B --head_type symbol_abstraction_head --step_size 10 --random_control --random_times_per_step $random_times --adaptive_step_size --rule $rule --prompt_file_for_causal_scores_exp "datasets/cma_scores/llama31_70B/symbol_abstraction_head/base_${rule}_prompt_for_cma_100.txt" --token_pos_list 4 10 --activation_name z --min_valid_sample_num $min_valid_sample_num  --load_generation_config --do_shuffle --seed $seed --prompt_num 1000 --low_prob_threshold 0.9 --n_devices 2 --token_set_file datasets/llama31_70B_correct_common_tokens_0.9_1378.txt  --in_context_example_num 2 --eval_metric gen_acc

rule=ABB # ABA or ABB
### symbolic induction head
python ablation.py --model_type Llama-3.1-70B --head_type symbolic_induction_head --step_size 2 --adaptive_step_size --rule $rule --prompt_file_for_causal_scores_exp "datasets/cma_scores/llama31_70B/symbolic_induction_head/base_${rule}_prompt_for_cma_100.txt" --token_pos_list -1 --activation_name z --min_valid_sample_num $min_valid_sample_num --load_generation_config  --do_shuffle --seed $seed --prompt_num 1000 --low_prob_threshold 0.9 --n_devices 2 --token_set_file datasets/llama31_70B_correct_common_tokens_0.9_1378.txt  --in_context_example_num 2 --eval_metric gen_acc
python ablation.py --model_type Llama-3.1-70B --head_type symbolic_induction_head --step_size 2 --control --adaptive_step_size --rule $rule --prompt_file_for_causal_scores_exp "datasets/cma_scores/llama31_70B/symbolic_induction_head/base_${rule}_prompt_for_cma_100.txt" --token_pos_list -1 --activation_name z --min_valid_sample_num $min_valid_sample_num  --load_generation_config  --do_shuffle --seed $seed --prompt_num 1000 --low_prob_threshold 0.9 --n_devices 2 --token_set_file datasets/llama31_70B_correct_common_tokens_0.9_1378.txt  --in_context_example_num 2 --eval_metric gen_acc
python ablation.py --model_type Llama-3.1-70B --head_type symbolic_induction_head --step_size 10 --random_control --random_times_per_step $random_times --adaptive_step_size --rule $rule --prompt_file_for_causal_scores_exp "datasets/cma_scores/llama31_70B/symbolic_induction_head/base_${rule}_prompt_for_cma_100.txt" --token_pos_list -1 --activation_name z --min_valid_sample_num $min_valid_sample_num  --load_generation_config --do_shuffle --seed $seed --prompt_num 1000 --low_prob_threshold 0.9 --n_devices 2 --token_set_file datasets/llama31_70B_correct_common_tokens_0.9_1378.txt  --in_context_example_num 2 --eval_metric gen_acc

### retrieval head
python ablation.py --model_type Llama-3.1-70B --head_type retrieval_head --step_size 2 --adaptive_step_size --rule $rule --prompt_file_for_causal_scores_exp "datasets/cma_scores/llama31_70B/retrieval_head/base_${rule}_prompt_for_cma_100.txt" --token_pos_list -1 --activation_name z --min_valid_sample_num $min_valid_sample_num --load_generation_config  --do_shuffle --seed $seed --prompt_num 1000 --low_prob_threshold 0.9 --n_devices 2 --token_set_file datasets/llama31_70B_correct_common_tokens_0.9_1378.txt  --in_context_example_num 2 --eval_metric gen_acc
python ablation.py --model_type Llama-3.1-70B --head_type retrieval_head --step_size 2 --control --adaptive_step_size --rule $rule --prompt_file_for_causal_scores_exp "datasets/cma_scores/llama31_70B/retrieval_head/base_${rule}_prompt_for_cma_100.txt" --token_pos_list -1 --activation_name z --min_valid_sample_num $min_valid_sample_num  --load_generation_config  --do_shuffle --seed $seed --prompt_num 1000 --low_prob_threshold 0.9 --n_devices 2 --token_set_file datasets/llama31_70B_correct_common_tokens_0.9_1378.txt  --in_context_example_num 2 --eval_metric gen_acc
python ablation.py --model_type Llama-3.1-70B --head_type retrieval_head --step_size 10 --random_control --random_times_per_step $random_times --adaptive_step_size --rule $rule --prompt_file_for_causal_scores_exp "datasets/cma_scores/llama31_70B/retrieval_head/base_${rule}_prompt_for_cma_100.txt" --token_pos_list -1 --activation_name z --min_valid_sample_num $min_valid_sample_num  --load_generation_config --do_shuffle --seed $seed --prompt_num 1000 --low_prob_threshold 0.9 --n_devices 2 --token_set_file datasets/llama31_70B_correct_common_tokens_0.9_1378.txt  --in_context_example_num 2 --eval_metric gen_acc
