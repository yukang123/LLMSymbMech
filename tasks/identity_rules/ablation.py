#####################################
import os 
import pyrootutils
root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=True)

from utils import HF_TOKEN, HF_HOME
os.environ["HF_TOKEN"] = HF_TOKEN # configure the User Access Token to authenticate to the Hub
os.environ["HF_HOME"] = HF_HOME  ## set the cache directory for Hugging Face 
#####################################

import logging

logger: logging.Logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from utils import LINE_SEP, plot, HEAD_ACTIVATIONS, set_seed, vocab_dict, get_model_id_family


from transformers import AutoTokenizer, GenerationConfig
import torch
import argparse
import random
import numpy as np
from tqdm import tqdm
import transformer_lens
from collections import defaultdict 
import seaborn as sns
import matplotlib.pyplot as plt
from functools import partial
import pandas as pd


def get_head_list(head_type):

    ## Load the causal mediation scores of each attention head for a specified head type 
    score_dict = {
        "symbol_abstraction_head": "datasets/cma_scores/llama31_70B/symbol_abstraction_head/causal_scores.pt",  #f"causal_mediation_results_full/meta-llama/Llama-3.1-70B_ungroup_gqa/exp_swap_1_2_change_rule_swap_1_2_question/base_rule_AVG/z_seed{tag}/logit/sample_num_20_0.0/group_heads_False/token_pos_[5, 11]/logits_diff_ch_avg_ABA_ABB_mean.pt",
        "symbolic_induction_head": "datasets/cma_scores/llama31_70B/symbolic_induction_head/causal_scores.pt", #f"causal_mediation_results_full/meta-llama/Llama-3.1-70B_ungroup_gqa/exp_swap_1_2_change_rule_swap_1_2_question/base_rule_AVG/z_seed{tag}/logit/sample_num_20_0.0/group_heads_False/token_pos_[-1]/logits_diff_ch_avg_ABA_ABB_mean.pt",
        "retrieval_head": "datasets/cma_scores/llama31_70B/retrieval_head/causal_scores.pt" #f"causal_mediation_results_full/meta-llama/Llama-3.1-70B_ungroup_gqa/exp_swap_1_2_question/base_rule_AVG/z_seed{tag}/logit/sample_num_20_0.0/group_heads_False/token_pos_[-1]/logits_diff_ch_avg_ABA_ABB_mean.pt"
    }

    file = score_dict[head_type]
    avg_score = torch.load(file, map_location="cpu", weights_only=True).cpu().numpy()

    n_layers = avg_score.shape[0]
    n_heads = avg_score.shape[1]

    layer_rank_head_list = np.argsort(avg_score, axis=1) ## rank the heads in each layer based on causal scores, from low to high
    indices = np.argsort(avg_score.reshape(-1))[::-1] ## rank all the heads from high to low
    layer_indices = indices // n_heads
    head_indices = indices % n_heads
    sorted_head_list = list(zip(layer_indices, head_indices)) ## get the (layer_idx, head_idx) for each head

    return sorted_head_list, layer_rank_head_list 

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, nargs="?", help="random seed")
    parser.add_argument("--rule", type=str, default="ABA", help="rule type")
    parser.add_argument("--head_type", type=str, default="symbol_abstraction_head", help="head type")

    ### load the model
    parser.add_argument("--model_type", type=str, default="Llama-3.1-70B", help="model type")
    parser.add_argument("--device_map", type=str, default="cpu", help="device map (cpu or auto)")
    parser.add_argument("--device", type=str, default="cuda", help="device")
    parser.add_argument("--n_devices", type=int, default=2, help="number of devices. please set to 2 for Llama-3.1-70B and Qwen2,5-72B while 1 for others")
    parser.add_argument("--fold_ln", action="store_true", help="fold layer normalization weights into the weights of the preceeding layer")

    #### build the prompt
    parser.add_argument("--prompt_num", type=int, default=None, help="the number of prompts, which will be used to filter out the correct prompts for ablation")
    parser.add_argument("--in_context_example_num", type=int, default=2, help="in-context example number")
    parser.add_argument("--sep_symbol", type=str, default="^", help="separator symbol")
    parser.add_argument("--do_shuffle", action="store_true")
    parser.add_argument(
        "--token_set_file", type=str, default="datasets/llama31_70B_correct_common_tokens_0.9_1378.txt", #None, 
        help="the file of token sets which form sequences of rules ABA and ABB on which the model could make correct predictions, each line is a set of (2*N) tokens separated by space, 'A_1 B_1 ... A_N B_N', e.g., 'la li te to hi ha' (N-1) in-context examples."
    )
    
    ### parameters for generation
    parser.add_argument("--eos_token", type=str, default=None, help="eos token") # will set as "\n" if not specified
    parser.add_argument("--max_new_tokens", type=int, default=10, help="max new tokens")
    parser.add_argument("--sample_size", type=int, default=4, help="sample size per prompt during generation")
    parser.add_argument("--load_generation_config", action="store_true")
    parser.add_argument("--generation_config_name", type=str, default=None, help="generation config file name, used when loading generation config from file")


    ### parameters for ablation
    parser.add_argument("--activation_name", default="z", type=str)
    parser.add_argument("--token_pos_list", nargs="+", default=[-1], type=int)
    parser.add_argument("--control", action="store_true", help="control group, ablate the same number of heads which have the lowest causal mediation scores in each layer")
    parser.add_argument("--random_control", action="store_true", help="random control group, randomly ablate the same number of heads among all the attention heads")
    parser.add_argument("--random_times_per_step", type=int, default=10, help="the number of random sampling trials at each cumulative step, only used for random control")
    parser.add_argument("--min_valid_sample_num", default=-1, type=int, help="minimum number of valid prompt pairs on which model made correct predictions, used for causal mediation score calculation")

    parser.add_argument("--eval_metric", type=str, default="gen_acc", choices=["gen_acc", "ans_prob"], help="evaluation metric for the model performance, gen_acc: generation accuracy, ans_prob: the probability of the correct answer")
    parser.add_argument("--low_prob_threshold", default=0.9, type=float)
    parser.add_argument("--start_head_idx", type=int, default=0, help="the index of the first head to ablate")
    parser.add_argument("--end_head_idx", type=int, default=None, help="the index of the last head to ablate")
    parser.add_argument("--step_size", type=int, default=1, help="step size for ablation")
    parser.add_argument("--prompt_file_for_causal_scores_exp", type=str, default=None, help="prompts which were used for calculating the causal scores, these prompts will not be used for ablation")

    parser.add_argument("--adaptive_step_size", action="store_true", help="adaptive step size for ablation")
    parser.add_argument("--patch_all_token_pos", action="store_true", help="patch all token positions")
    parser.add_argument("--log_dir", type=str, default="results/identity_rules/ablation", help="log directory")
    parser.add_argument("--verbose", action="store_true", help="verbose")

    args = parser.parse_args()
    return args


def generate_prompts(
        args, tokenizer, vocab_file, sep_symbol = "^", token_sets=None, 
        do_shuffle=False
    ):
    prompts = []
    correct_ans_list = []

    assert os.path.exists(vocab_file), f"Vocabulary file {vocab_file} does not exist."
    with open(vocab_file, "r") as f:
        vocab_list = [l.rstrip() for l in f.readlines()]
    # assert exp_swap_1_2_change_rule or causal_swap_1_2_within_instance or exp_keep_1_2_change_rule or exp_swap_1_2_question

    if do_shuffle:
        np.random.shuffle(vocab_list)
        if token_sets is not None:
            np.random.shuffle(token_sets)

    while len(prompts) < args.prompt_num: # or len(ABB_prompts) < args.prompt_num:

        ## generate the prompt
        ## 1. get the tokens for each in-context example
        if token_sets is None:
            tokens = random.sample(vocab_list, k =(args.in_context_example_num + 1) * 2)
        else:
            tokens = token_sets.pop(0) 

       ## 2. build the prompt
        prompt = ""
        for j in range(args.in_context_example_num):
            if args.rule == "ABA":
                tj = [tokens[j*2], tokens[j*2 + 1], tokens[j*2]]
            elif args.rule == "ABB":
                tj = [tokens[j*2], tokens[j*2 + 1], tokens[j*2 + 1]]
            p_j = f"{sep_symbol}".join(tj)
            prompt += p_j + "\n"
        ## add the question
        prompt += tokens[-2] + f"{sep_symbol}" + tokens[-1] + f"{sep_symbol}"
        if args.rule == "ABA":
            correct_ans = tokens[-2]
        elif args.rule == "ABB":
            correct_ans = tokens[-1]
        
        ### sanity check to make sure that the tokens will not be grouped with separation symbols in the tokenization. ###
        proc_tokens = tokenizer.tokenize(prompt)
        token_num = len(proc_tokens)
        if token_num != 3 * 2 * args.in_context_example_num + 4:
            logger.info("====================================================")
            logger.info(f"Invalid prompt: {prompt}, token_num: {token_num}, expected: {3 * 2 * args.in_context_example_num + 4}")
            continue
        else:
            if np.sum(~np.isin(proc_tokens[1::2], [sep_symbol, tokenizer.tokenize("\n")[0]])):
                logger.info("====================================================")
                logger.info(f"Invalid prompt: {prompt}, tokens: {proc_tokens}")
                continue
        
        if prompt not in prompts:
            prompts.append(prompt)
            correct_ans_list.append(correct_ans)
        else:
            logger.info("====================================================")
            logger.info(f"Repeated prompt: {prompt}")

    return prompts, correct_ans_list

def generate_response_eval(
        model: transformer_lens.HookedTransformer, 
        args,
        input_ids,
        eos_token_id,
        tokenizer: AutoTokenizer,
        correct_ans,   
        prompt,
        **generation_kwargs
    ):
    """
        Generate responses and evaluate the generation accuracy. Capable of adding hooks to the model (e.g., patching activation) during generation.
    """
    input_ids = input_ids.repeat(args.sample_size, 1)
    generated_content = model.generate(
        input_ids, 
        max_new_tokens=args.max_new_tokens, 
        eos_token_id = eos_token_id, 
        stop_at_eos=True,
        return_type="input",
        verbose=False,
        **generation_kwargs
    )
    generated_ids = generated_content
    text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    acc = 0
    response_ids = generated_ids[:, input_ids.shape[1]:]
    correct_num = 0
    total_num = 0
    for response_id in response_ids:
        eos_sign = False
        ## find the first eos token
        for ri, r in enumerate(response_id): 
            if r in eos_token_id:
                r_mask = ri
                eos_sign = True
                break
        if not eos_sign: ## response does not contain the eos token
            r_mask = len(response_id)
            logger.warning("--------------------------------------------------")
            logger.warning(f"Response does not contain the eos token: {response_id}")

        valid_response_id = response_id[:r_mask] ## remove the eos token
        gen_ans = tokenizer.decode(valid_response_id, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        acc += (gen_ans == correct_ans)
        correct_num += (gen_ans == correct_ans)
        total_num += 1
        if r_mask > 1:  # generated answers include multiple tokens
            multipe_tokens = tokenizer.convert_ids_to_tokens(response_id)
            logger.warning("--------------------------------------------------")
            logger.warning(f"Question: {prompt}")
            logger.warning(f"Response contains multiple tokens: {gen_ans}, correct ans: {correct_ans}, multiple tokens: {multipe_tokens}")
    acc = acc / args.sample_size
    return correct_num, total_num, acc

def ablate_head(
        model: transformer_lens.HookedTransformer, 
        ranked_head_list, ## ranked head list from top to bottom
        input_ids, 
        ans, ## the correct answer
        token_pos=-1, 
        layer_ranked_heads=None, ## from bottom to top in each layer
        total_layers=80, total_heads=64, 
        activation_name="z",
        start_head_idx=0, 
        end_head_idx=None,
        step_size=1,
        control=False,
        random_control=False,
        random_times_per_step=5,
        seed = 42, 
        adaptive_step_size=False,
        patch_all_token_pos=False, 

    ):
    
    assert not random_control or not control, "random control and control cannot be both True"
    assert input_ids is not None
    token_len = input_ids.shape[-1]
    assert token_len > 1, "token length should be larger than 1"
    assert ranked_head_list is not None
    if patch_all_token_pos:
        token_pos = range(token_len) 
        logger.info(f"patch all token positions: {token_pos}")

    def replace_head_activation_hook(activation, hook, head_idx=None, head_dict=None, token_len=17):
        layer_idx = hook.layer()
        assert head_dict is not None
        try:
            ablate_heads = head_dict[layer_idx]
        except:
            return
        if control:
            ablate_heads = layer_ranked_heads[layer_idx][:len(ablate_heads)]

        if len(ablate_heads) == 0:
            return
        
        if activation.shape[1] == token_len:  
            for head_idx in ablate_heads:
                activation[:, token_pos, head_idx, :] = 0
    

    fn = model.run_with_hooks 
    ans_prob_ablated_all = []

    i_ = start_head_idx
    random_i_ = 0
    act_tag = lambda cache_: cache_.endswith("hook_" + activation_name)
    top_num_list = []
    end_head_idx = len(ranked_head_list) if end_head_idx is None else end_head_idx
    tqdm_bar = tqdm()

    random_command = random.Random(seed)
    while i_ <= end_head_idx: 
        if random_control:
            full_head_set = [(i, j) for i in range(total_layers) for j in range(total_heads)]
            selected_heads = random_command.sample(full_head_set, i_)
            random_i_ += 1
        else:  
            selected_heads = ranked_head_list[:i_]
        selected_heads_dict = defaultdict(list)
        for layer_idx, head_idx in selected_heads:
            selected_heads_dict[layer_idx].append(head_idx)
        
        top_num_list.append(len(selected_heads))

        logits = fn(
            input_ids,
            fwd_hooks=[(
                act_tag,
                partial(replace_head_activation_hook, head_dict=selected_heads_dict, token_len=token_len)
            )]
        )
        ans_id = model.to_single_token(ans)
        ans_prob_ablated = torch.softmax(logits, dim=-1)[0,-1,ans_id].cpu()
        ans_prob_ablated_all.append(ans_prob_ablated)


        if adaptive_step_size: ## [TODO] make it more general
            if not control and not random_control: ## only for the non-control group
                if i_ > 3500: ## after ablating more than 3500 heads, the probability of the correct answer will be very low, so we can increase the step size to speed up the ablation process
                    step_size = 8
            else:
                if len(ranked_head_list) - i_ < 500:
                    step_size = 4 #1

        del logits

        torch.cuda.empty_cache()
        if random_control and random_i_ < random_times_per_step:
            continue
        else:
            i_ += step_size
            tqdm_bar.update(step_size)
            random_i_ = 0
    
    return (
        torch.tensor(ans_prob_ablated_all),
        top_num_list
    )


def cumulative_ablation(
        model: transformer_lens.HookedTransformer, 
        tokenizer, 
        prompt_list, 
        correct_ans_list, 
        head_type,
        activation_name, 
        save_folder, 
        token_pos=-1, 
        device="cuda", 
        ranked_head_list=None,
        layer_ranked_heads=None,
        start_head_idx=0,
        end_head_idx=None,
        step_size=1,
        control=False,
        random_control=False,
        random_times_per_step=5,
        prompt_for_causal_scores=None,
        seed = 42,
        adaptive_step_size=False,
        eval_metric="gen_acc",
        min_valid_sample_num=-1, 
        low_prob_threshold=0.9, 
        eos_token_id=None, 
        patch_all_token_pos=False,
        **generation_kwargs
    ):
    """
    Conduct cumulative ablation on the model for a list of prompts.

    Args:
        model (HookedTransformer): The model to be ablated.
        tokenizer (AutoTokenizer): The tokenizer for the model.
        prompt_list (list): List of prompts to be ablated.
        correct_ans_list (list): List of correct answers corresponding to the prompts.
        head_type (str): Type of the head to be ablated.
        activation_name (str): Name of the activation to be ablated.
        save_folder (str): Folder to save the results.
        token_pos (int or list): Position of the token to be ablated. If -1, the last token will be ablated.
        device (str): Device to run the model.
        ranked_head_list (list): List of heads ranked by causal mediation scoresf from top to bottom.
        layer_ranked_heads (dict): Dictionary of heads ranked by causal mediation scores from bottom to top in each layer.
        start_head_idx (int): Index of the first head to ablate
        end_head_idx (int): Index of the last head to ablate. If None, all heads will be ablated.
        step_size (int): Step size for ablation.
        control (bool): Whether to use control group for ablation.
        random_control (bool): Whether to use random control group for ablation.
        random_times_per_step (int): Number of random times per step for random control group.
        prompt_for_causal_scores (list): List of prompts which were used for causal scores calculation
        seed (int): Random seed for reproducibility.
        adaptive_step_size (bool): Whether to use ununiform step size for ablation.
        eval_metric (str): Evaluation metric to use for ablation, default is "gen_acc".
        min_valid_sample_num (int): Minimum number of valid samples required for ablation.
        low_prob_threshold (float): Low probability threshold for filtering prompts.
        eos_token_id (int): End-of-sequence token ID.
        patch_all_token_pos (bool): [Optional] Whether to patch all token positions.
        **generation_kwargs: Additional keyword arguments for generation.

    """

    assert activation_name in HEAD_ACTIVATIONS, f"activation_name {activation_name} not in {HEAD_ACTIVATIONS}"
    total_layers = model.cfg.n_layers
    total_heads = model.cfg.n_heads
    real_ans_prob_patched_list = []
    valid_num = 0
    selected_prompt_list = []
    for idx_, prompt in enumerate(prompt_list):
        logger.info(f"Processing {idx_}th sample")
        ans = correct_ans_list[idx_]

        if prompt_for_causal_scores is not None and prompt in prompt_for_causal_scores:
            print(f"Skip the base prompt {prompt} which was used for calculating the causal scores")
            continue

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = inputs.input_ids
        
        if eval_metric == "gen_acc":
            ## evaluate the model performance through generation accuracy
            _, _, acc = generate_response_eval(
                model, args, input_ids, eos_token_id, tokenizer, ans, prompt,
                **generation_kwargs
            )
            if acc < low_prob_threshold:
                logger.warning(f"low acc {acc} < {low_prob_threshold}")
                continue

        logits = model(input_ids)
        ans_real_prob = torch.softmax(logits, dim=-1)[0,-1,model.to_single_token(ans)] 
        if (eval_metric == "ans_prob") and (ans_real_prob < low_prob_threshold):
            logger.info(f"true ans {ans_real_prob} lower than {low_prob_threshold}")
            continue
        
        selected_prompt_list.append(prompt)
        results = ablate_head(
            model, 
            ranked_head_list,
            input_ids, 
            ans,
            token_pos,
            layer_ranked_heads=layer_ranked_heads,
            total_layers=total_layers, 
            total_heads=total_heads,
            activation_name=activation_name,
            start_head_idx=start_head_idx,
            end_head_idx=end_head_idx,
            step_size=step_size,
            control=control,
            random_control=random_control,
            random_times_per_step=random_times_per_step,
            seed=seed,
            adaptive_step_size=adaptive_step_size,
            patch_all_token_pos=patch_all_token_pos,
        )
        real_ans_prob_patched_all, top_num_list = results
        real_ans_prob_patched_list.append(real_ans_prob_patched_all.float())

        valid_num += 1

        if min_valid_sample_num != -1 and valid_num >= min_valid_sample_num: ## run ablation on enough valid samples on which the model could generate the correct answers
            break

    if valid_num == 0:
        logger.warning("No samples to patch")
        return
    
    save_folder = os.path.join(save_folder, head_type, f"sample_num_{valid_num}_{eval_metric}_{low_prob_threshold}")
    os.makedirs(save_folder, exist_ok=True)
    with open(os.path.join(save_folder, f"prompt_{valid_num}.txt"), "w") as f:
        for prompt in selected_prompt_list:
            f.write(f"{LINE_SEP}" + prompt + "\n")

    token_pos_tag = f"token_pos_{token_pos}" 
    if patch_all_token_pos:
        token_pos_tag = f"token_pos_all" 
    
    ablation_type = f"control_{control}_random_control_{random_control}_{random_times_per_step}"
    sub_save_folder = os.path.join(save_folder, ablation_type, token_pos_tag)
    os.makedirs(sub_save_folder, exist_ok=True)
    mean_real_ans_prob_patched = torch.stack(real_ans_prob_patched_list, dim=0).mean(dim=0)
    torch.save(torch.stack(real_ans_prob_patched_list, dim=0), os.path.join(sub_save_folder, "real_ans_prob_patched.pt")) 
    np.save(os.path.join(sub_save_folder, f"top_num_list.npy"), np.array(top_num_list))
    plot_curve(mean_real_ans_prob_patched.cpu().numpy(), top_num_list, sub_save_folder, metric_name=f"Correct Answer Probability")


def plot_curve(metric_list, top_num_list, save_folder, metric_name):

    df = pd.DataFrame({
        "head_k": top_num_list,
        "metric": metric_list
    })
    plt.figure()

    sns.lineplot(data=df, x="head_k", y="metric", errorbar="se")
    plt.xlabel("Number of Heads")
    plt.ylabel(metric_name)
    plt.title(f"Effect of Top-K Heads on {metric_name}")
    os.makedirs(save_folder, exist_ok=True)
    plt.savefig(os.path.join(save_folder, f"{metric_name}_vs_top_k_heads.png"))

def main(args):
    set_seed(args.seed)

    # get the model id and family
    model_id, model_family =  get_model_id_family(args.model_type) 
    vocab_file = vocab_dict[model_family] ## get the vocabulary file path
    logger.info(f"model type: {args.model_type}, model id: {model_id}, vocab file: {vocab_file}")

    #################################################################
    # 0. Load the model and tokenizer, 
    # and specify the generation config which will be used to filter out the correct prompts for CMA
    #################################################################
    logger.info(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    torch.set_grad_enabled(False)
    
    ## specify the sampling config for generation
    if args.load_generation_config: ## load the generation config for the model
        ### Use the generation config saved in ./generation_config/ folder
        ### OR GET THE GENERATION CONFIG FILE BY setting args.save_generation_config to True when running the tasks/identity_rules/eval.py 
        ### OR
        # model = AutoModelForCausalLM.from_pretrained(
        #     model_id, device_map=args.device_map, 
        #     torch_dtype=torch.bfloat16
        # )
        # model.generation_config.save_pretrained("./generation_config/", config_file_name=f"{args.model_type}.json")
        ###############
        ###############
        generation_config_name = args.generation_config_name if args.generation_config_name is not None else f"{args.model_type}.json"
        generation_config = GenerationConfig.from_pretrained("./generation_config/", config_file_name=generation_config_name)
        logger.info(f"loading generation config from ./generation_config/{generation_config_name}")
        generation_kwargs = {
            "top_k": generation_config.top_k,
            "top_p": generation_config.top_p,
            "do_sample": generation_config.do_sample,
            "temperature": generation_config.temperature,
        }
        generation_kwargs.update({
        "freq_penalty": generation_config.repetition_penalty
        })
    else: ## Some default generation settings for multinomial sampling
        logger.warning("Using default multinomial sampling generation config [Not recommended]")
        generation_kwargs = {
            "top_k": 50,
            "top_p": 0.9,
            "do_sample": True,
            "temperature": 0.6,
        }
    logger.info(f"generation_kwargs: {generation_kwargs}")

    ## Load the model
    try:
        model = transformer_lens.HookedTransformer.from_pretrained(
            model_id, 
            device_map=args.device_map, 
            device=args.device,
            n_devices=args.n_devices,
            torch_dtype=torch.bfloat16,
            tokenizer=tokenizer,
            ########################################
            fold_ln=args.fold_ln, # False, ## Whether to fold in the LayerNorm weights to the subsequent linear layer. This does not change the computation. (applicable to RMSNorm)
            center_writing_weights=False, ## Because we are using RMSNorm, the writing weights will not be centered
            center_unembed=False,               
            ) 
        device = model.cfg.device
    except Exception as e:
        logger.error(f"Error: {e}")
        logger.error("Failed to load the model. Please check the model id")
        return
    model_remark = ""
    # Ungroup the grouped query attention, **NECESSARY** for CMA on keys/values, which are not done in the paper 
    logger.info("ungroup the keys/values which were grouped in the grouped query attention, this is necessary for CMA on keys/values (not done in the paper)")
    model.set_ungroup_grouped_query_attention(True)
    logger.info(f"set use_past_kv_cache to False to avoid the error occurring in the generation with ungrouped keys/values")
    generation_kwargs["use_past_kv_cache"] = False # set to False to avoid the error occurring in the generation with ungrouped keys/values
    model_remark += "_ungroup_gqa"

    # add new eos token based on the prompt format [Useful for Generation]
    # the prompt format is like: "la^li^la\nte^to^te\nha^hi^", where the eos token should be "\n"
    eos_token_id = tokenizer.eos_token_id
    vocab = tokenizer.get_vocab()
    if args.eos_token is None:
        args.eos_token = "\n"
    add_eos_token = tokenizer.convert_ids_to_tokens(tokenizer.encode(args.eos_token)[-1])
    logger.info(f"eos_token: {args.eos_token}, add_eos_token: {add_eos_token}")
    add_eos_token_dict = {v:vocab[v] for v in vocab if v.startswith(add_eos_token)}
    add_eos_token_id = list(add_eos_token_dict.values())
    if type(eos_token_id) != list:
        logger.info(f"eos_token_id: {eos_token_id}, eos: {tokenizer.convert_ids_to_tokens(eos_token_id)}")
        eos_token_id = [eos_token_id]
    eos_token_id.extend(add_eos_token_id)

    ## create the result folder
    activation_name = args.activation_name
    remark = f"{model_id}{model_remark}/rule_{args.rule}/{activation_name}_seed_{args.seed}_shuffle_{args.do_shuffle}"
    save_folder = os.path.join(args.log_dir, remark)
    os.makedirs(save_folder, exist_ok=True)
    logger.info(f"save_folder: {save_folder}")

    #################################################################
    #### 1. Build the prompt dataset ####
    #################################################################
    logger.info(f"generate prompts.... (SEP symbol: {args.sep_symbol})")
    assert args.prompt_num is not None
    token_sets = None
    if args.token_set_file is not None:
        assert os.path.exists(args.token_set_file)
        with open(args.token_set_file, "r") as f:
            token_sets = f.readlines()
            token_sets = [t.rstrip().split(" ") for t in token_sets]
    prompts, correct_ans_list = generate_prompts(
        args, tokenizer, vocab_file, sep_symbol = args.sep_symbol, token_sets=token_sets, 
        do_shuffle=args.do_shuffle,
    )
    with open(os.path.join(save_folder, f"input_prompts_{len(prompts)}.txt"), "w") as f:
        for p in prompts:
            f.write(f"{LINE_SEP}" + p + "\n")
    
    ## load the prompts which were used for calculating the causal scores, these prompts will not be used for ablation
    if args.prompt_file_for_causal_scores_exp is not None:
        with open(args.prompt_file_for_causal_scores_exp, "r") as f:
            lines = f.read()
            lines = lines.split(LINE_SEP)   

            assert lines[0] == ""
            lines = lines[1:] 
        prompt_for_causal_scores = [line.strip() for line in lines]
    else:
        prompt_for_causal_scores = None

    #################################################################
    #### 2. Get the head list ranked by the causal mediation scores ####
    #################################################################

    ranked_head_list, layer_rank_head_dict  = get_head_list(args.head_type)


    ##################################################################
    #### 3. Conduct cumulative ablation ####
    ##################################################################

    token_pos = args.token_pos_list
    prepend_bos = model.cfg.default_prepend_bos
    logger.info(f"prepend_bos: {prepend_bos}")
    if prepend_bos:
        token_pos = [pos+1 if pos != -1 else pos for pos in token_pos]
    min_valid_sample_num = args.min_valid_sample_num if args.min_valid_sample_num is not None else -1
    low_prob_threshold = args.low_prob_threshold #0.9

    logger.info(f"Ablate the activations {activation_name} at pos {token_pos} (min_valid_sample_num: {min_valid_sample_num})")
    logger.info(f"control: {args.control}, random_control: {args.random_control}, random_times_per_step: {args.random_times_per_step}")
    cumulative_ablation(
        model, tokenizer, prompts, correct_ans_list, args.head_type,
        activation_name, 
        save_folder, token_pos, 
        device=device,
        min_valid_sample_num=min_valid_sample_num, 
        ranked_head_list=ranked_head_list, layer_ranked_heads=layer_rank_head_dict,
        start_head_idx=args.start_head_idx,
        end_head_idx=args.end_head_idx,
        step_size=args.step_size,
        control=args.control,
        random_control=args.random_control,
        random_times_per_step=args.random_times_per_step, 
        eval_metric=args.eval_metric,
        eos_token_id=eos_token_id, 
        low_prob_threshold=low_prob_threshold, 
        prompt_for_causal_scores=prompt_for_causal_scores,
        seed=args.seed,
        adaptive_step_size=args.adaptive_step_size,
        patch_all_token_pos=args.patch_all_token_pos,
        **generation_kwargs
    )


if __name__ == "__main__":
    args = get_args()
    main(args)