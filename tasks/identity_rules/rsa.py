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

from utils import LINE_SEP, plot, set_seed, vocab_dict, get_model_id_family, get_head_list

from transformers import AutoTokenizer
import transformer_lens
import torch
import argparse
import random
import numpy as np
from tqdm import tqdm
from collections import defaultdict 
from scipy.stats import pearsonr
import gc

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, nargs="?", help="random seed")

    ## parameters for the model
    parser.add_argument("--model_type", type=str, default="Llama-3.1-70B", help="model type")   
    parser.add_argument("--device_map", type=str, default="cpu", help="device map")
    parser.add_argument("--device", type=str, default="cuda", help="device")
    parser.add_argument("--n_devices", type=int, default=1, help="number of devices")
    parser.add_argument("--fold_ln", action="store_true", help="fold layer normalization")

    ## parameters for the prompts
    parser.add_argument("--base_rule", type=str, default="ABA", help="base rule")
    parser.add_argument("--in_context_example_num", type=int, default=2, help="in-context example number")
    parser.add_argument(
        "--token_set_file", type=str, default="datasets/llama31_70B_correct_common_tokens_0.9_1378.txt", #None, 
        help="the file contains token sets that formed sequences of rules ABA and ABB, with the model predicting both correctly, each line is a set of (2*N) tokens separated by space, 'A_1 B_1 ... A_N B_N', e.g., 'la li te to hi ha' (N-1) in-context examples."
    )
    parser.add_argument("--add_swap_1_2_question", action="store_true")
    parser.add_argument("--prompt_num", type=int, default=None, help="prompt number")
    parser.add_argument("--sep_symbol", type=str, default="^", help="separator symbol")
    parser.add_argument("--do_shuffle", action="store_true")


    ## parameters for rsa
    parser.add_argument("--cmp_with_abstract", action="store_true", help="compare with abstract variables")
    parser.add_argument("--cmp_with_token_id", action="store_true", help="compare with token id")
    parser.add_argument("--sel_pos_list", nargs="+", help="selected position list", type=int)
    parser.add_argument("--act_list", nargs="+", default=["z"], help="the list of activations to be recorded")
    parser.add_argument("--use_attn_result", action="store_true", help="whether to split w_o into sub-blocks and apply each subblock on corresponding attention head's output") 
    parser.add_argument("--low_sim", type=float, default=0.0, help="low similarity")
    parser.add_argument("--high_sim", type=float, default=1.0, help="high similarity")

    parser.add_argument("--only_for_significant_heads", action="store_true", help="conduct rsa only for significant heads")
    parser.add_argument("--head_type", type=str, default="symbol_abstraction_head", choices=["symbol_abstraction_head", "symbolic_induction_head", "retrieval_head"], help="the head type to be analyzed")

    parser.add_argument("--start_layer_idx", type=int, default=0, help="start layer index")
    parser.add_argument("--end_layer_idx", type=int, default=None, help="end layer index")
    parser.add_argument("--start_head_idx", type=int, default=0, help="start head index")
    parser.add_argument("--end_head_idx", type=int, default=None, help="end head index")
    parser.add_argument("--transpose", action="store_true", help="transpose the feature matrix before calculating similarity")
    parser.add_argument("--plot_hand_code", action="store_true", help="plot hand-coded")
    parser.add_argument("--plot_similarity", action="store_true", help="plot similarity")
    parser.add_argument("--plot_rsa", action="store_true", help="plot RSA")
    parser.add_argument("--save_similarity", action="store_true", help="save similarity")
    parser.add_argument("--verbose", action="store_true", help="verbose")
    parser.add_argument("--log_dir", type=str, default="results/identity_rules/rsa", help="log directory")

    args = parser.parse_args()
    return args

def generate_prompts(
        args, tokenizer, vocab_file, sep_symbol = "^", token_sets=None, 
        add_swap_1_2_question=False,
        base_rule="ABA", do_shuffle=False
    ):
    prompts = []
    correct_ans_list = []
    rule_group_list = []
    # causal_ans_list = []

    assert os.path.exists(vocab_file)
    with open(vocab_file, "r") as f:
        vocab_list = [l.rstrip() for l in f.readlines()] 

    if do_shuffle:
        np.random.shuffle(vocab_list)
        if token_sets is not None:
            np.random.shuffle(token_sets)

    while len(prompts) < args.prompt_num: 

        ## generate the prompt
        ## 1. get the tokens for each in-context example
        if token_sets is None:
            tokens = random.sample(vocab_list, k =(args.in_context_example_num + 1) * 2)
        else:
            tokens = token_sets.pop(0) # la li te to hi ha 


        base_example_list = []
        exp_example_list = []

        for idx_ in range(args.in_context_example_num):

            if base_rule == "ABA":
                base_tokens = [tokens[idx_*2], tokens[idx_*2+1], tokens[idx_*2]]  ## la li la
                exp_tokens = [tokens[idx_*2+1], tokens[idx_*2], tokens[idx_*2]] ## li la la
                exp_rule = "ABB"

            elif base_rule == "ABB":
                base_tokens = [tokens[idx_*2], tokens[idx_*2+1], tokens[idx_*2+1]] ## la li li
                exp_tokens = [tokens[idx_*2+1], tokens[idx_*2], tokens[idx_*2+1]] ## li la li
                exp_rule = "ABA"

            base_example = f"{sep_symbol}".join(base_tokens)
            base_example_list.append(base_example)

            exp_example = f"{sep_symbol}".join(exp_tokens)
            exp_example_list.append(exp_example)

        # incomplete query (question) for the last in-context example
        base_question_tokens = [tokens[-2], tokens[-1]] # hi ha
        base_ans_index = 0 if base_rule == "ABA" else 1
        base_ans = base_question_tokens[base_ans_index]

        base_question = f"{sep_symbol}".join(base_question_tokens) + f"{sep_symbol}"
        base_example_list.append(base_question)
        base_prompt = "\n".join(base_example_list)

        exp_question_tokens = [tokens[-1], tokens[-2]] # ha hi
        exp_question = f"{sep_symbol}".join(exp_question_tokens) + f"{sep_symbol}" 
        exp_example_list.append(exp_question)
        exp_prompt = "\n".join(exp_example_list)
        exp_ans_index = 1 - base_ans_index
        exp_ans = exp_question_tokens[exp_ans_index]

        prompt_group = [base_prompt, exp_prompt]
        ans_group = [base_ans, exp_ans] 
        rule_group = [base_rule, exp_rule]
        ####################################
        ## Abstract context pair:
        ## base: la li la te to te hi ha (ABA)
        ## exp: li la la to te te ha hi (ABB)
        ####################################

        if add_swap_1_2_question: 
            ##################################
            ## for each context (prompt) in the abstract context pair,
            ## swap the first and sencond tokens in the question to form a new context 
            ## hi ha -> ha hi

            ## add_base_prompt: la li la te to te ha hi (swap 1/2 in the question for base)
            ## add_exp_prompt: li la la to te te hi ha (swap 1/2 in the question for exp)
            #################################
            add_base_question_tokens = base_question_tokens[::-1]
            add_base_question = f"{sep_symbol}".join(add_base_question_tokens) + f"{sep_symbol}"
            add_base_ans = add_base_question_tokens[base_ans_index]

            base_example_list[-1] = add_base_question
            add_base_prompt = "\n".join(base_example_list)
            prompt_group.append(add_base_prompt)
            ans_group.append(add_base_ans)

            add_exp_question_tokens = exp_question_tokens[::-1]
            add_exp_question = f"{sep_symbol}".join(add_exp_question_tokens) + f"{sep_symbol}"
            add_exp_ans = add_exp_question_tokens[exp_ans_index]

            exp_example_list[-1] = add_exp_question
            add_exp_prompt = "\n".join(exp_example_list)
            prompt_group.append(add_exp_prompt)
            ans_group.append(add_exp_ans)

            rule_group = rule_group * 2

        ### sanity check ###
        pass_tag = True
        for prompt in prompt_group:
            proc_tokens = tokenizer.tokenize(prompt)
            token_num = len(proc_tokens)
            if token_num != 3 * 2 * args.in_context_example_num + 4:
                if args.verbose:
                    logger.info("====================================================")
                    logger.info(f"Invalid prompt: {prompt}, token_num: {token_num}, expected: {3 * 2 * args.in_context_example_num + 4}")
                pass_tag = False
                break
            else:
                # if np.sum([1 for k_ in proc_tokens[1::2] if k_ not in [sep_symbol, tokenizer.tokenize("\n")[0]]]):
                if np.sum(~np.isin(proc_tokens[1::2], [sep_symbol, tokenizer.tokenize("\n")[0]])):
                    if args.verbose:
                        logger.info("====================================================")
                        logger.info(f"Invalid prompt: {prompt}, tokens: {proc_tokens}")
                    pass_tag = False
                    break
        if not pass_tag:
            continue

        if prompt_group not in prompts:
            prompts.append(prompt_group)
            correct_ans_list.append(ans_group)
            rule_group_list.append(rule_group)
        else:
            if args.verbose:
                logger.info("====================================================")
                logger.info(f"Repeated prompt: {prompt}")

    return prompts, correct_ans_list, rule_group_list 


def main(args):
    set_seed(args.seed)
    act_list = args.act_list ## ["z", "v", "rot_k", "rot_q", "result", "k", "q"]

    model_id, model_family =  get_model_id_family(args.model_type) 
    vocab_file = vocab_dict[model_family] ## get the vocabulary file path
    logger.info(f"model type: {args.model_type}, model id: {model_id}, vocab file: {vocab_file}")


    #################################################################
    # 0. Load the model and tokenizer, 
    # and specify the generation config which will be used to filter out the correct prompts for CMA
    #################################################################

    logger.info(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, token = HF_TOKEN) 
    torch.set_grad_enabled(False)
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
    except Exception as e:
        logger.error(f"Error: {e}")
        logger.error("Failed to load the model. Please check the model id")
        return
    device = model.cfg.device

    low_sim = args.low_sim
    high_sim = args.high_sim

    
    if "result" in args.act_list or args.use_attn_result: 
        # whether to split w_o into sub-blocks and apply each sublock on corresponding attention head's output, 
        # so the final output of the activation block w_o * [z[0] ... z[63]] is rewritten into the summation of w_o[i] * z[i] over all attention heads
        # And conduct rsa on the activations after applying sub-blocks of w_o on each attention head's output, i.e., "result" in act_list
        logger.info("Splitting w_o into sub-blocks and applying each subblock on corresponding attention head's output")
        model.set_use_attn_result(True)

    logger.info("ungroup the keys/values which were grouped in the grouped query attention, this is necessary for CMA on keys/values (not done in the paper)")
    model.set_ungroup_grouped_query_attention(True)
    model_remark = "_ungroup_gqa"

    prompt_group_size = 2
    causal_remark = "abstract"
    if args.add_swap_1_2_question:
        causal_remark += "_add_swap_1_2_question"
        prompt_group_size *= 2 # 4

    folder_remark = f"{model_id}{model_remark}/in_context_example_{args.in_context_example_num}/{causal_remark}_base_rule_{args.base_rule}/prompt_{args.prompt_num}_seed_{args.seed}_shuffle_{args.do_shuffle}" 
    save_folder = os.path.join(args.log_dir, folder_remark)
    logger.info(f"save_folder: {save_folder}")
    os.makedirs(save_folder, exist_ok=True)

    #################################################################
    #### 1. Build the prompt dataset ####
    #################################################################
    logger.info(f"generate prompts.... (SEP symbol: {args.sep_symbol})")
    assert args.prompt_num is not None
    ## Strongly recommend to use the token set file that contains the tokens that the model can predict correctly
    if args.token_set_file is not None:
        assert os.path.exists(args.token_set_file)
        logger.info(f"Using token set file: {args.token_set_file}, which contains the tokens that the model can predict correctly")
        with open(args.token_set_file, "r") as f:
            token_sets = f.readlines()
            token_sets = [t.rstrip().split(" ") for t in token_sets]
    else:
        logger.warning("No token set file is provided, random tokens will be used, which model may not predict correctly")
        token_sets = None
    prompts, correct_ans_list, rule_group_list = generate_prompts(
        args, tokenizer, vocab_file,
        sep_symbol=args.sep_symbol, token_sets=token_sets,
        add_swap_1_2_question=args.add_swap_1_2_question,
        base_rule=args.base_rule,
        do_shuffle=args.do_shuffle,
        )
    for idx_ in range(len(prompts[0])):
        with open(os.path.join(save_folder, f"set_{idx_}_input_prompts_{args.prompt_num}.txt"), "w") as f:
            for p in prompts:
                f.write(f"{LINE_SEP}" + p[idx_] + "\n")

    with open(os.path.join(save_folder, f"ans_{args.prompt_num}.txt"), "w") as f:
        for idx_, ans_pair in enumerate(correct_ans_list):
            f.write(" ".join(ans_pair) + "\n")

    with open(os.path.join(save_folder, f"rule_{args.prompt_num}.txt"), "w") as f:
        for idx_, rule_pair in enumerate(rule_group_list):
            f.write(" ".join(rule_pair) + "\n")
    logger.info(f"# total prompts: {len(prompts)}")


    #################################################################
    #### 2. Collect Activations ####
    #################################################################
    all_act_dict = {"ABA": defaultdict(list), "ABB": defaultdict(list)}
    cache_input_ids_dict = defaultdict(list)
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    prepend_bos = True
    for i, prompt_pair in enumerate(tqdm(prompts)):
        for j_ in range(len(prompt_pair)):
            prompt = [prompt_pair[j_]]
            correct_ans = correct_ans_list[i][j_]
            rule_j = rule_group_list[i][j_] ## "ABA" or "ABB"
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            input_ids = inputs.input_ids
            cache_input_ids = input_ids[0].unsqueeze(0)    

            cache_input_ids_cp = cache_input_ids.clone()
            correct_ans_id = tokenizer.convert_tokens_to_ids(correct_ans)
            assert correct_ans_id is not None
            cache_input_ids_cp[:, -1] = correct_ans_id
            cache_input_ids_dict[rule_j].append(cache_input_ids_cp)

            logits, cache = model.run_with_cache(cache_input_ids)
            act_dict = {} 
            for act in act_list:
                act_values = [cache[act, layer_idx].cpu() for layer_idx in range(n_layers)]
                if act == "pattern" or act == "attn_scores":
                    act_dict[act] = torch.stack(act_values, dim=1) # batch_size x layers x x heads x token_num x token_num
                else:
                    act_dict[act] = torch.stack(act_values, dim=2) # batch_size x seq_len x n_layers x head_index x d_head 
            for act in act_dict.keys():
                # (for same prompt, the activations for each layer are the same)
                act_dict[act] = act_dict[act].mean(dim=0) # (seq_len x n_layers x head_index x d_head) or (seq_len x d_model)
                all_act_dict[rule_j][act].append(act_dict[act])

            del logits
            del cache
            del act_values
            del act_dict 
            gc.collect()
            torch.cuda.empty_cache()

    logger.info("Finish recording activations")
    del model
    torch.cuda.empty_cache()

    if len(list(all_act_dict.keys())) == 0:
        logger.warning("No activations are recorded")
        return

    if args.only_for_significant_heads:
        head_list, head_weight_score = get_head_list(args.head_type)
        weighted_avg_sim = None
        weighted_rsa_corr = 0
        weighted_sum = 0
        logger.info(f"Focusing on significant heads for {args.head_type}: {head_list}")

    cache_input_ids_dict["ABA"] = torch.concat(cache_input_ids_dict["ABA"], dim=0)    
    cache_input_ids_dict["ABB"] = torch.concat(cache_input_ids_dict["ABB"], dim=0)
    ## key, query, value, output
    for act in all_act_dict["ABA"].keys():
        all_act_dict["ABA"][act] = torch.stack(all_act_dict["ABA"][act], dim=0) 
        all_act_dict["ABB"][act] = torch.stack(all_act_dict["ABB"][act], dim=0) 
        logger.info(f"Aggregate activations for {act} ABA: {all_act_dict['ABA'][act].shape} ABB: {all_act_dict['ABB'][act].shape}")
        assert args.cmp_with_abstract or args.cmp_with_token_id 
        sel_pos_list = args.sel_pos_list
        if prepend_bos:
            sel_pos_list = [pos + 1 for pos in sel_pos_list]
        all_act = torch.concat([all_act_dict["ABA"][act], all_act_dict["ABB"][act]], dim=0).float()
        # relative position for each item in the in-context example
        # pos_table = {   
        #     i_: 1 for i_ in range(1, all_act.shape[1], 6)
        # }
        # pos_table.update({   
        #     i_: 2 for i_ in range(3, all_act.shape[1], 6)
        # })
        # pos_table.update({
        #     i_: 3 for i_ in range(5, all_act.shape[1], 6)
        # })
        # pos_table[all_act.shape[1]-1] = 3 

        #################################################################
        #### 3. Getting Expected Similarity Matrix ####
        #################################################################
        logger.info("Getting Expected Similarity Matrix Based on Abstract Variables or Literal Tokens...")
        ### table of abstract variables for each token in the sequences of rules ABA and ABB
        ## 1: A; 2: B
        abs_table = {   
            i_: 1 for i_ in range(1, all_act.shape[1], 6)
        }
        abs_table.update({   
            i_: 2 for i_ in range(3, all_act.shape[1], 6)
        })
        ABA_abs_table = abs_table.copy()
        ABA_abs_table.update({
            i_: 1 for i_ in range(5, all_act.shape[1], 6)
        })            
        ABA_abs_table[all_act.shape[1]-1] = 1
        ABB_abs_table = abs_table.copy()
        ABB_abs_table.update({
            i_: 2 for i_ in range(5, all_act.shape[1], 6)
        })            
        ABB_abs_table[all_act.shape[1]-1] = 2

        ### all_act: (2 * prompt_num) x seq_len x n_layers x n_heads x d_head
        total_prompt_num = all_act.shape[0]

        n_layers = all_act.shape[-3]
        n_heads = all_act.shape[-2]
        
        if args.cmp_with_token_id:

            sel_pos_list_ABA = sel_pos_list
            sel_pos_list_ABB = sel_pos_list

            ABA_input_ids = cache_input_ids_dict["ABA"][:, sel_pos_list_ABA]
            ABB_input_ids = cache_input_ids_dict["ABB"][:, sel_pos_list_ABB]

            ABA_act = all_act[:total_prompt_num//2, sel_pos_list_ABA, ...]
            ABB_act = all_act[total_prompt_num//2:, sel_pos_list_ABB, ...]

            all_input_ids = torch.concat([ABA_input_ids, ABB_input_ids], dim=0)
            all_act_sel = torch.concat([ABA_act, ABB_act], dim=0)
            if args.transpose:
                all_act_sel = all_act_sel.swapaxes(0,1)
                all_input_ids = all_input_ids.swapaxes(0,1)

            all_input_ids = all_input_ids.reshape(-1)

            hand_code_sim_matrix = all_input_ids.unsqueeze(1) == all_input_ids.unsqueeze(0)
            hand_code_sim_matrix = hand_code_sim_matrix.float()
            hand_code_sim_matrix = hand_code_sim_matrix * high_sim + (1 - hand_code_sim_matrix) * low_sim

            all_act_sel = all_act_sel.reshape(-1, *all_act_sel.shape[2:]) ## (2 * prompt_num * len(sel_pos_list)) x n_layers x n_heads x d_head
            token_sub_save_folder = os.path.join(save_folder, f"{act}_RSA_with_token_ids/token_pos_{'_'.join([str(pos) for pos in sel_pos_list])}_transpose_{args.transpose}")


        elif args.cmp_with_abstract:
            
            ABA_abstract = torch.tensor([ABA_abs_table[pos] for pos in sel_pos_list])
            ABB_abstract = torch.tensor([ABB_abs_table[pos] for pos in sel_pos_list])

            ABA_abstract = ABA_abstract.unsqueeze(0).repeat(total_prompt_num//2, 1)
            ABB_abstract = ABB_abstract.unsqueeze(0).repeat(total_prompt_num//2, 1)

            all_abstract = torch.concat([ABA_abstract, ABB_abstract], dim=0)
            all_act_sel = all_act[:, sel_pos_list, ...]

            if args.transpose:
                all_act_sel = all_act_sel.swapaxes(0,1)
                all_abstract = all_abstract.swapaxes(0,1)

            all_abstract = all_abstract.reshape(-1)
            all_act_sel = all_act_sel.reshape(-1, *all_act_sel.shape[2:]) ## (2 * prompt_num * len(sel_pos_list)) x n_layers x n_heads x d_head

            hand_code_sim_matrix = all_abstract.unsqueeze(1) == all_abstract.unsqueeze(0)
            hand_code_sim_matrix = hand_code_sim_matrix.float()
            hand_code_sim_matrix = hand_code_sim_matrix * high_sim + (1 - hand_code_sim_matrix) * low_sim

            token_sub_save_folder = os.path.join(save_folder, f"{act}_RSA_with_abstract_variables/token_pos_{'_'.join([str(pos) for pos in sel_pos_list])}_transpose_{args.transpose}")

        os.makedirs(token_sub_save_folder, exist_ok=True)
        torch.save(hand_code_sim_matrix.cpu(), os.path.join(token_sub_save_folder, f"hand_code_sim_matrix.pt"))
        if args.plot_hand_code:
            plot(hand_code_sim_matrix.cpu().numpy(), token_sub_save_folder, metric_name=f"Hand Code Similarity Matrix High {high_sim} Low {low_sim}", num_for_one_rule=all_act_sel.shape[0]//2, xlabel_name="Index", ylabel_name="Index")

        #################################################################
        #### 3. Calculate the pairwise similarity for activations ####
        #################################################################
        logger.info("Calculating RSA...")
        rsa_correlation_matrix = np.zeros((n_layers, n_heads)) * np.nan
        start_layer_idx = args.start_layer_idx
        end_layer_idx = args.end_layer_idx if args.end_layer_idx is not None else n_layers
        start_head_idx = args.start_head_idx 
        end_head_idx = args.end_head_idx if args.end_head_idx is not None else n_heads

        for layer_idx in tqdm(range(start_layer_idx, end_layer_idx)):
            for head_idx in range(start_head_idx, end_head_idx):

                if args.only_for_significant_heads and (layer_idx, head_idx) not in head_list:
                    continue
                        
                feature = all_act_sel[:, layer_idx, head_idx, :]
                feature_sim = cal_similarity(feature)

                if args.plot_similarity:
                    plot(
                        feature_sim.cpu().numpy(), token_sub_save_folder, metric_name=f"Layer_{layer_idx}_Head_{head_idx}", 
                        num_for_one_rule=feature_sim.shape[0]//2 if not args.transpose else None, xlabel_name="Index", ylabel_name="Index",
                        fig_w=55, fig_h=35
                        )
                
                if args.save_similarity:
                    torch.save(feature_sim.cpu(), os.path.join(token_sub_save_folder, f"similarity_matrix_Layer_{layer_idx}_Head_{head_idx}.pt"))

                rsa_correlation = compare_two_similarity_matrix(feature_sim, hand_code_sim_matrix, include_diagnoal=False) 
                rsa_correlation_matrix[layer_idx, head_idx] = rsa_correlation
                if args.only_for_significant_heads:
                    weight = head_weight_score[layer_idx, head_idx]
                    if weighted_avg_sim is None:
                        weighted_avg_sim = feature_sim.cpu() * weight
                    else:
                        weighted_avg_sim += feature_sim.cpu() * weight
                    weighted_rsa_corr += rsa_correlation * weight.item()
                    weighted_sum += weight.item()

        np.save(os.path.join(token_sub_save_folder, f"rsa_correlation_matrix_high_sim_{high_sim}_low_sim_{low_sim}.npy"), rsa_correlation_matrix)
        if args.plot_rsa:
            plot(
                rsa_correlation_matrix, token_sub_save_folder, metric_name=f"RSA_Correlation_Matrix_{act}_High_{high_sim}_Low_{low_sim}",
            )
    
        if args.only_for_significant_heads:
            weighted_avg_sim = weighted_avg_sim / weighted_sum
            weighted_rsa_corr /= weighted_sum
            torch.save(weighted_avg_sim.cpu(), os.path.join(token_sub_save_folder, f"{args.head_type}_significant_head_{len(head_list)}_weighted_sim.pt"))
            plot(weighted_avg_sim.cpu().numpy(), token_sub_save_folder, f"{args.head_type} Weighted Average Similarity Matrix", xlabel_name="", ylabel_name="")
            logger.info(f"<{args.head_type}|{act}> Weighted RSA correlation across all {len(head_list)} significant heads: {weighted_rsa_corr}")
            with open(os.path.join(token_sub_save_folder, f"significant_head_weighted_rsa_corr.txt"), "a") as f:
                f.write(f"{args.head_type} ({len(head_list)}): {weighted_rsa_corr}\n")


def cal_similarity(features):
    norm_features = torch.nn.functional.normalize(features, p=2, dim=-1)
    norm_features = torch.movedim(norm_features, 0, -2)
    sim = torch.matmul(norm_features, norm_features.transpose(-1, -2))
    return sim

def compare_two_similarity_matrix(real_matrix, exp_matrix, include_diagnoal=False):
    ## Calculate the Pearson correlation coefficient between the lower triangular part of two similarity matrices
    tril_indices = torch.tril_indices(exp_matrix.shape[0], exp_matrix.shape[1], offset=-1 if not include_diagnoal else 0)
    exp_matrix_tril_values = exp_matrix[tril_indices[0], tril_indices[1]].cpu().numpy()
    real_matrix_tril_values = real_matrix[tril_indices[0], tril_indices[1]].cpu().numpy()
    psr, _ = pearsonr(exp_matrix_tril_values, real_matrix_tril_values)
    return psr


if __name__ == "__main__":
    args = get_args()
    main(args)