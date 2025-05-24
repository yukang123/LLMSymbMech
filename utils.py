import numpy as np
import torch
import random
import os
import seaborn as sns
import matplotlib.pyplot as plt
import logging

logger: logging.Logger = logging.getLogger(__name__)

HF_HOME = "~/.cache/huggingface" ## set cache folder for huggingface on your choice like this "/scratch/gpfs/userid/.cache". ignore this if using the default dir
HF_TOKEN = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" # replace with your own HF token; create one under Access Tokens in your HF account
LINE_SEP = "---------------------\n"
HEAD_ACTIVATIONS = ["q", "k", "v", "z", "pattern", "attn_scores", "rot_q", "rot_k"]

vocab_dict = {
    "gpt2": "datasets/vocab/gpt2_english_vocab.txt",
    "Llama-3": "datasets/vocab/llama31_english_vocab.txt", 
    # For later version of Llama-3, please check whether the vocabulary changes (using datasets/get_vocab.py)
    "Qwen2.5": "datasets/vocab/qwen25_english_vocab.txt",
    "gemma-2": "datasets/vocab/gemma2_english_vocab.txt",
}

model_family_dict = {
    "gpt2": ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"],
    "Llama-3": ["Llama-3.1-70B", "Llama-3.1-8B"],
    "Qwen2.5": ["Qwen2.5-7B", "Qwen2.5-14B", "Qwen2.5-32B", "Qwen2.5-72B"],
    "gemma-2": ["gemma-2-2b", "gemma-2-9b", "gemma-2-27b"], 
}

## the file recording the significant heads for Llama-3.1-70B
## a n_layers x n_heads (80 * 64) tensor 
## the entries for significant heads are the causal mediation scores of the head while the entries for non-significant heads are 0
llama31_70B_significant_head_dict = {
    "symbol_abstraction_head": "datasets/cma_scores/llama31_70B/symbol_abstraction_head/significant_heads.pt", #"causal_mediation_results_full/meta-llama/Llama-3.1-70B_ungroup_gqa/exp_swap_1_2_change_rule_swap_1_2_question/base_rule_AVG/z_seed_0_1_2_3_4_shuffle_True/logit/sample_num_20_0.0/group_heads_False/token_pos_[5, 11]/null_trial_num_5000_quantile_0.05_discret_3000/significant_heads_quantile_0.05_threshold_0.13_scores.pt",
    "symbolic_induction_head": "datasets/cma_scores/llama31_70B/symbolic_induction_head/significant_heads.pt", #"causal_mediation_results_full/meta-llama/Llama-3.1-70B_ungroup_gqa/exp_swap_1_2_change_rule_swap_1_2_question/base_rule_AVG/z_seed_0_1_2_3_4_shuffle_True/logit/sample_num_20_0.0/group_heads_False/token_pos_[-1]/null_trial_num_5000_quantile_0.05_discret_3000/significant_heads_quantile_0.05_threshold_0.18_scores.pt",
    "retrieval_head": "datasets/cma_scores/llama31_70B/retrieval_head/significant_heads.pt", #"causal_mediation_results_full/meta-llama/Llama-3.1-70B_ungroup_gqa/exp_swap_1_2_question/base_rule_AVG/z_seed_0_1_2_3_4_shuffle_True/logit/sample_num_20_0.0/group_heads_False/token_pos_[-1]/null_trial_num_5000_quantile_0.05_discret_3000/significant_heads_quantile_0.05_threshold_0.22_scores.pt"
}

def get_head_list(head_type, head_dict=llama31_70B_significant_head_dict):
    file_name = head_dict[head_type]
    weight_score = torch.load(file_name, map_location="cpu")
    weight_score = weight_score.cpu()
    weight_score = torch.clip(weight_score, min=0)
    places = torch.where(weight_score > 0)
    head_list = [(places[0][i].item(), places[1][i].item()) for i in range(len(places[0]))]
    return head_list, weight_score

def get_model_id_family(model_type):
    if "Llama-3" in model_type:
        model_id = f"meta-llama/{model_type}"
        model_family = "Llama-3"
    ### Qwen 2.5
    elif "Qwen2.5" in model_type:
        model_id = f"Qwen/{model_type}"
        model_family = "Qwen2.5"
    ### gemma-2
    elif "gemma-2" in model_type:
        model_id = f"google/{model_type}"
        model_family = "gemma-2"
    ### gpt2
    elif "gpt2" in model_type:
        model_id = model_type
        model_family = "gpt2"
    else:
        raise ValueError(f"Unknown model family for {model_type}. Please configurate the model_family (model_id) and download the corresponding vocab file (using datasets/get_vocab.py).")
    return model_id, model_family


def set_seed(seed): #, fully_deterministic=False):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # if fully_deterministic: ## only for fully reproducible results, but will make sampling much slower and take more memory
    #     os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
    #     torch.use_deterministic_algorithms(True)

def load_prompts(prompt_file, rule, sep_symbol, in_context_num, sample_num=None, by_permutation=False): ## TODO: test this function
    prompts = []
    correct_ans_list = []
    with open(prompt_file, "r") as f:
        lines = f.read()
        lines = lines.split(LINE_SEP)
        assert lines[0] == ""
        lines = lines[1:]
        for l in lines:
            prompt = l.rstrip()
            prompts.append(prompt)

            examples = prompt.split("\n")
            sample_tb_filled = examples[-1].split(sep_symbol)
            assert len(examples) == in_context_num + 1
            assert len(sample_tb_filled) == 3

            token_sets = sample_tb_filled[:-1]
            for i in range(in_context_num):
                in_context_sample = examples[i].split(sep_symbol)
                if rule == "ABA":
                    assert in_context_sample[0] == in_context_sample[2], f"The prompts stored in the file {prompt_file} do not follow rule '{rule}'"
                elif rule == "ABB":
                    assert in_context_sample[1] == in_context_sample[2], f"The prompts stored in the file {prompt_file} do not follow rule '{rule}'"
                token_sets.extend(in_context_sample)
        
            if rule == "ABA":
                correct_ans = sample_tb_filled[0]
            elif rule == "ABB":
                correct_ans = sample_tb_filled[1]
            correct_ans_list.append(correct_ans)

    if (sample_num is not None) and (sample_num != -1): ## NOT RECOMMENDED
        sample_num = min(sample_num, len(prompts))
        logger.info(f"Randomly select {sample_num} prompts")
        sel_index = random.sample(range(len(prompts)), sample_num)
        prompts = [prompts[i] for i in sel_index]
        correct_ans_list = [correct_ans_list[i] for i in sel_index]
    return prompts, correct_ans_list


def get_token_set(prompts, sep_symbol="^"):
    ## get the set of unique tokens in each prompt
    ## la li la te to te ha hi ha -> la li te to ha hi 
    all_token_set = []
    for p in prompts:
        all_tokens = p.split("\n")
        token_set = []
        for token in all_tokens:
            token_set.extend(token.split(sep_symbol)[:2])
        all_token_set.append(" ".join(token_set))
    return all_token_set

def plot(result_array, save_folder, metric_name="Test Accuracy", num_for_one_rule=None, xlabel_name="Head Index", ylabel_name="Layer Index", pos_labels=None, xpos_labels=None, ypos_labels=None, invert_y_axis=False, fig_w=30, fig_h = 22, annot_labels=None, colormap="viridis"):
    annot_font_size = 8    
    colorbar_font_size = 24
    linewidth = 5
    orig_rcParams = plt.rcParams.copy()
    plt.rcParams.update({
    # 'font.size': 12,           # default font size
    'xtick.labelsize': 18,     # xtick label size
    'ytick.labelsize': 18,     # ytick label size
    'axes.labelsize': 40,      # xlabel, ylabel fontsize
    'axes.titlesize': 40,      # title fontsize
    # 'cbar.labelsize': 12       # colorbar fontsize
    })
    fig = plt.figure(figsize=(fig_w, fig_h))
    # if j_ == 0:
    #     sns.heatmap(heatmap_dict[act].cpu().numpy(), annot=True, fmt=".2f", cbar=True, cmap = "RdYlBu_r", vmin=-1.0, vmax=1.0, annot_kws={"size": annot_font_size})
    # else:
    
    if annot_labels is None:
        ax = sns.heatmap(result_array, annot=True, fmt=".2f", cbar=True, cmap = colormap, annot_kws={"size": annot_font_size}, mask=(np.isnan(result_array))) ##, mask=(result_array == -1))
    else:
        ax = sns.heatmap(result_array, annot=annot_labels, fmt="", cbar=True, cmap = colormap, annot_kws={"size": annot_font_size}, mask=(np.isnan(result_array)))
    remark = ""
    if invert_y_axis:
        ax.invert_yaxis()
        remark = "_invert_y_axis"

    if num_for_one_rule is not None:
        assert not invert_y_axis
        plt.axvline(x=num_for_one_rule, color='black', linewidth=linewidth)
        plt.axhline(y=num_for_one_rule, color='black', linewidth=linewidth)

    colorbar = fig.axes[-1] 
    colorbar.tick_params(labelsize=colorbar_font_size)
    os.makedirs(save_folder, exist_ok=True)
    plt.xlabel(xlabel_name)
    plt.ylabel(ylabel_name)

    xpos_labels = xpos_labels if xpos_labels is not None else pos_labels
    ypos_labels = ypos_labels if ypos_labels is not None else pos_labels

    if xpos_labels is not None:
        assert not invert_y_axis
        plt.xticks(np.arange(len(xpos_labels)) + 0.5, xpos_labels)
    
    if ypos_labels is not None:
        assert not invert_y_axis
        plt.yticks(np.arange(len(ypos_labels)) + 0.5, ypos_labels)

    plt.title(f"{metric_name}")
    plt.savefig(os.path.join(save_folder, f"{metric_name}{remark}_heatmap.png"))
    plt.close(fig)

    plt.rcParams.update(orig_rcParams)
    return
        



