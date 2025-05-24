#####################################
import os 
import pyrootutils
root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=True)

from utils import HF_TOKEN, HF_HOME
os.environ["HF_TOKEN"] = HF_TOKEN # configure the User Access Token to authenticate to the Hub
os.environ["HF_HOME"] = HF_HOME  ## set the cache directory for Hugging Face 
#####################################
## implementation of CMA method (Section 3.1 and Alogorithm 1 in https://arxiv.org/pdf/2502.20332?)
## context_type: abstract (symbol abstraction heads/ symbolic induction heads) or token (retrieval heads)
## notations: (patching from base context to exp context (c2 -> c1))
# base context (c2): base_rule/base_prompt/base_ans (y_c2)
# exp context (c1): exp_rule/exp_prompt/exp_ans (y_c1)
# patched exp context (c1*): expected answer after patching activations from c2 to c1, causal_ans (y_c1*)
######################################

from typing import (Any, Callable, List, Tuple, Union)
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
from functools import partial
import transformer_lens.utils as utils

class CustomHookedTransformer(transformer_lens.HookedTransformer):
    def generate_with_hooks(
        self, 
        *model_args: Any,
        fwd_hooks: List[Tuple[Union[str, Callable], Callable]] = [],
        bwd_hooks: List[Tuple[Union[str, Callable], Callable]] = [],
        reset_hooks_end: bool = True,
        clear_contexts: bool = False,
        **model_kwargs: Any
    ):
        """
            Support implementing forward (e.g. patching activations) and backward hooks during generation. 
            Adapted from generate() and run_with_hooks() in transformer_lens.
        """
        if len(bwd_hooks) > 0 and reset_hooks_end:
            logging.warning(
                "WARNING: Hooks will be reset at the end of generate_with_hooks. This removes the backward hooks before a backward pass can occur."
            )
  
        with self.hooks(fwd_hooks, bwd_hooks, reset_hooks_end, clear_contexts) as hooked_model:
            return hooked_model.generate(*model_args, **model_kwargs)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, nargs="?", help="random seed")

    ### load the model
    parser.add_argument("--model_type", type=str, default="Llama-3.1-70B", help="model type")
    parser.add_argument("--device_map", type=str, default="cpu", help="device map (cpu or auto)")
    parser.add_argument("--device", type=str, default="cuda", help="device")
    parser.add_argument("--n_devices", type=int, default=2, help="number of devices. please set to 2 for Llama-3.1-70B and Qwen2,5-72B while 1 for others")
    parser.add_argument("--fold_ln", action="store_true", help="fold layer normalization weights into the weights of the preceeding layer")

    #### build the prompt (context) pairs 
    parser.add_argument("--context_type", type=str, default="abstract", help="whether to build abstract context (symbol abstraction heads/ symbolic induction heads) or token context (retrieval heads)", choices=["abstract", "token"])
    parser.add_argument("--base_rule", type=str, default="ABA", help="rule for base prompt in prompt pairs", choices=["ABA", "ABB"])
    parser.add_argument("--prompt_num", type=int, default=None, help="the number of prompt pairs to be generated")
    parser.add_argument("--in_context_example_num", type=int, default=2, help="in-context example number")
    parser.add_argument("--sep_symbol", type=str, default="^", help="separator symbol")
    parser.add_argument("--do_shuffle", action="store_true", help="whether to shuffle the vocabulary list for generating prompts")

    ### parameters for generation
    parser.add_argument("--eos_token", type=str, default=None, help="eos token") # will set as "\n" if not specified
    parser.add_argument("--max_new_tokens", type=int, default=10, help="max new tokens")
    parser.add_argument("--sample_size", type=int, default=4, help="sample size per prompt during generation")
    parser.add_argument("--load_generation_config", action="store_true")
    parser.add_argument("--generation_config_name", type=str, default=None, help="generation config file name, used when loading generation config from file")

    ## parameters for patching the activations
    parser.add_argument("--activation_name", default="z", type=str, help="activation name, 'z' for individual attention head output or 'attn_out' for the whole attention block output (after aggregating all attention heads through w_o) in each layer")
    parser.add_argument("--patch_mlp_out", action="store_true", help="[necessary and only applicable when activation_name == 'attn_out'] whether to patch the MLP output so that we can replace all the information added into the residual stream in each layer, i.e., attention block output and MLP output")
    parser.add_argument("--token_pos_list", nargs="+", default=[-1], type=int, help="token positions where we do patching")
    parser.add_argument("--min_valid_sample_num", default=-1, type=int, help="minimum number of valid prompt pairs on which model made correct predictions, used for causal mediation score calculation")
    parser.add_argument("--eval_metric", default="gen_acc", help="evaluation metric for filtering prompts", choices=["gen_acc", "ans_prob"])
    parser.add_argument("--low_prob_threshold", default=0.9, type=float, help="threshold for filtering low probability/accuracy samples")
    parser.add_argument("--ungroup_grouped_query_attention", action="store_true", help="ungroup the grouped query attention")

    parser.add_argument("--log_dir", type=str, default="results/identity_rules/cma", help="log directory")
    parser.add_argument("--generate", action="store_true", help="whether to measure the causal effects by actually generating the responses")
    parser.add_argument("--group_heads", action="store_true", help="whether to patch the activations of grouped heads which share the keys/values (GQA) at the same time")
    parser.add_argument("--verbose", action="store_true", help="verbose")

    args = parser.parse_args()
    return args


def generate_prompts(
        args, tokenizer, vocab_file, context_type, sep_symbol = "^", 
        base_rule="ABA", do_shuffle=False,
    ):
    prompts = []
    correct_ans_list = []
    causal_ans_list = []

    assert os.path.exists(vocab_file)
    with open(vocab_file, "r") as f:
        vocab_list = [l.rstrip() for l in f.readlines()]

    if do_shuffle:
        np.random.shuffle(vocab_list)
    assert context_type in ["abstract", "token"], f"Invalid context type: {context_type}"

    while len(prompts) < args.prompt_num:

        ## generate the prompt
        ## 1. get the tokens for each in-context example
        tokens = random.sample(vocab_list, k =(args.in_context_example_num + 1) * 2)

        base_example_list = []
        exp_example_list = []

        for idx_ in range(args.in_context_example_num):

            if base_rule == "ABA":
                base_tokens = [tokens[idx_*2], tokens[idx_*2+1], tokens[idx_*2]]  ## la li la
                if context_type == "abstract": ## abstract context
                    # swap 1st item and 2nd item in each in-context example while keeping the third item the same
                    # in the context pair, the third token is the same but the symbol which the third item represents is different
                    exp_tokens = [tokens[idx_*2+1], tokens[idx_*2], tokens[idx_*2]] ## li la la
                else:
                    exp_tokens = base_tokens ## la li li

            elif base_rule == "ABB":
                base_tokens = [tokens[idx_*2], tokens[idx_*2+1], tokens[idx_*2+1]] ## la li li
                if context_type == "abstract":
                    exp_tokens = [tokens[idx_*2+1], tokens[idx_*2], tokens[idx_*2+1]] ## li la li
                else:
                    exp_tokens = base_tokens  ## la li li

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

        exp_question_tokens = [tokens[-1], tokens[-2]] # ha hi # swap the 1/2 item in the last incomplete query (question)

        exp_question = f"{sep_symbol}".join(exp_question_tokens) + f"{sep_symbol}" 
        exp_example_list.append(exp_question)
        exp_prompt = "\n".join(exp_example_list)

        causal_ans = None


        if context_type == "abstract": ## applicable for symbol abstraction heads and symbolic induction heads
            ## before causal mediation, the rules are different;
            ## if the attention head output embedding represents abstract symbol, 
            ## after patching the embedding from base to exp (at the third item in each in-context example),
            ## it should induce the rule change of exp to the same as base

            ## example: ABA -> ABB (could be the other way around as well, i.e., ABB to ABA)
            ## base: la li la te to te hi ha (ABA), the third item "la" binds to symbol "A"
            ## exp: li la la to te te ha hi (ABB), the third item "la" binds to symbol "B"
            ## patch from base -> exp: 
            ## the expected answer of patched exp should correspond to the symbol "A", which is "ha"

            ## please refer to the paper for more detailed explanations.
            exp_ans_index = 1 - base_ans_index  ## different rule
            causal_ans_index = base_ans_index ### the expected answer after causal mediation

        else: ## applicable for retrieval heads
            # assert exp_swap_1_2_question == True ## for retreival head, the question should be swapped
            ## the rule is the same for prompt pairs but the correct answers for the last query are different

            ## example: ABA (also applicable for ABB)
            ## la li la te to te hi ha
            ## la li la te to te ha hi

            ## If output embedding represents the literal token,
            ## when swap the output embeddings at the last token, the answer should be the same as the base case

            exp_ans_index = base_ans_index
            causal_ans = base_ans 

        exp_ans = exp_question_tokens[exp_ans_index]
        if causal_ans is None:
            causal_ans = exp_question_tokens[causal_ans_index]

        ### sanity check ###
        pass_tag = True
        for prompt in [base_prompt, exp_prompt]:
            proc_tokens = tokenizer.tokenize(prompt)
            token_num = len(proc_tokens)
            if token_num != 3 * 2 * args.in_context_example_num + 4:
                if args.verbose:
                    logger.info("====================================================")
                    logger.info(f"Invalid prompt: {prompt}, token_num: {token_num}, expected: {3 * 2 * args.in_context_example_num + 4}")
                pass_tag = False
                break
            else:
                if np.sum(~np.isin(proc_tokens[1::2], [sep_symbol, tokenizer.tokenize("\n")[0]])):
                    if args.verbose:
                        logger.info("====================================================")
                        logger.info(f"Invalid prompt: {prompt}, tokens: {proc_tokens}")
                    pass_tag = False
                    break
        if not pass_tag:
            continue
        
        if (base_prompt, exp_prompt) not in prompts:
            prompts.append((base_prompt, exp_prompt))
            correct_ans_list.append((base_ans, exp_ans))
            causal_ans_list.append(causal_ans)
        else:
            if args.verbose:
                logger.info("====================================================")
                logger.info(f"Repeated prompt: {prompt}")

    return prompts, correct_ans_list, causal_ans_list

def cal_logit_prob_diff(logits, model: transformer_lens.HookedTransformer, causal_ans, original_ans):
    """
        calculate the logit differences between the expected answer of the patched context and the correct answer of the original context. 
    """
    causal_ans_id = model.to_single_token(causal_ans)
    original_ans_id = model.to_single_token(original_ans)
    logits_diff = logits[0, -1, causal_ans_id] - logits[0, -1, original_ans_id]
    return logits_diff

def generate_response_eval(
        model: CustomHookedTransformer, 
        args,
        input_ids,
        eos_token_id,
        tokenizer: AutoTokenizer,
        correct_ans,   
        prompt,
        fwd_hooks: List[Tuple[Union[str, Callable], Callable]] = [],
        **generation_kwargs
    ):
    """
        Generate responses and evaluate the generation accuracy. Capable of adding hooks to the model (e.g., patching activation) during generation.
    """
    input_ids = input_ids.repeat(args.sample_size, 1)
    generated_content = model.generate_with_hooks(
        input_ids, 
        max_new_tokens=args.max_new_tokens, 
        eos_token_id = eos_token_id, 
        stop_at_eos=True,
        return_type="input",
        verbose=False,
        fwd_hooks=fwd_hooks,
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
    
### [1] patching the activations in different layers x different heads at certain token positions
def ablate_head(
        model: CustomHookedTransformer, 
        base_cache, ## activation cache
        input_ids, ## input ids
        token_pos=-1, ## the token positions where activations are patched
        exp_logits_diff=None,
        # ans_1=None, ans_2=None,
        patched_exp_ans=None, exp_ans=None,
        total_layers=80, total_heads=64, group_size=1,
        activation_name="z", device="cuda", 
        ## whether to generate the responses while patching the activations
        generate=False, tokenizer=None, correct_ans=None, prompt=None, eos_token_id=None, 
        **generation_kwargs
        ):
    """
    Patch the activations in different layers and heads at certain token positions. Please refer to CMA section and Algorithm 1 in https://arxiv.org/pdf/2502.20332? for more details.
    
    Args:
        model: CustomHookedTransformer, the model to be patched.
        base_cache: ActivationCache, the activation cache from the base context (c2) in Algorithm 1 https://arxiv.org/pdf/2502.20332?.
        input_ids: Tensor, the input ids for the model.
        token_pos: int or list, the token positions where activations are patched.
        exp_logits_diff: 
            Tensor, 
            in the exp context (c1), the difference between the logits f(.) for the expected answer (y_c1*) in the patched context and the correct answer (y_c1) of the exp context, 
            i.e., \delta(f_c1) = f(c1)[y_c1*] - f(c1)[y_c1].
        patched_exp_ans: 
            str: The expected answer for the patched context c1*: y_c1* 
            After patching the activations, the answer for the patched context would change 
            according to our hypotheses about the representations of the activations (i.e., whether they represent abstract symbols or literal tokens).
        exp_ans: 
            str: The correct answer for the original context c1: y_c1.
        total_layers: int, the total number of layers in the model.
        total_heads: int, the total number of heads in the model.
        group_size: int, the group size for head splitting.
        activation_name: str, the name of the activation to be patched.
        device: str, the device to run the model on.

        ## whether to generate the responses and test the accuracy of y_c1 while patching the activations
        generate: bool, whether to actually generate responses while patching activations.
        tokenizer: PreTrainedTokenizer, the tokenizer to use for decoding.
        correct_ans: str, the correct answer string.
        prompt: str, the prompt string.
        eos_token_id: int, the end of sequence token id.
        **generation_kwargs: additional keyword arguments for generation.

    """
    ### [Important Function] ###
    def replace_head_activation_hook(activation, hook, head_idx, token_len=17, group_size=1):
        activation_tag = hook.name
        if activation.shape[1] == token_len:    
            rule_1_act_cache = base_cache[activation_tag][:, token_pos, head_idx * group_size: (head_idx+1) * group_size, :]
            activation[:, token_pos, head_idx * group_size: (head_idx+1) * group_size, :] = rule_1_act_cache

    assert input_ids is not None
    token_len = input_ids.shape[-1]
    assert token_len > 1, "token length should be larger than 1"

    if generate: ## whether to measure the causal effects by actually generating the responses
        correct_num_all = torch.zeros((total_layers, total_heads), device=device)
        total_num_all = torch.zeros((total_layers, total_heads), device=device)
        acc_all = torch.zeros((total_layers, total_heads), device=device)
    else:
        fn = model.run_with_hooks 
        logits_diff_change = torch.zeros((total_layers, total_heads), device=device)

    for layer_idx in tqdm(range(total_layers)):
        act_tag = utils.get_act_name(activation_name, layer_idx)

        for head_idx in range(total_heads):            
            if generate: ## generating the response while implementing activation patching during the run
                correct_num, total_num, acc = generate_response_eval(
                    model, args, input_ids, eos_token_id, tokenizer, correct_ans, prompt,
                    fwd_hooks=[(
                        act_tag,
                        partial(replace_head_activation_hook, head_idx = head_idx, token_len=token_len, group_size=group_size)
                    )], 
                    **generation_kwargs
                )
                correct_num_all[layer_idx, head_idx] = correct_num
                total_num_all[layer_idx, head_idx] = total_num
                acc_all[layer_idx, head_idx] = acc
            else:
                patched_logits = fn(
                    input_ids,
                    fwd_hooks=[(
                        act_tag,
                        partial(replace_head_activation_hook, head_idx = head_idx, token_len=token_len, group_size=group_size)
                    )]
                )
                patched_exp_logits_diff =  cal_logit_prob_diff(
                    patched_logits, model, causal_ans=patched_exp_ans, original_ans=exp_ans)
                logits_diff_change[layer_idx, head_idx] = patched_exp_logits_diff -  exp_logits_diff ## calculate the change in logits difference after patching the activations

    if generate:
        return correct_num_all, total_num_all, acc_all    
    else:
        return logits_diff_change

### [2] patching the activations in different layers x token positions
def ablate_layer(
        model: CustomHookedTransformer, 
        base_cache, 
        input_ids, 
        exp_logits_diff=None,
        # ans_1=None, ans_2=None,
        patched_exp_ans=None, exp_ans=None,
        total_layers=80, 
        activation_name="attn_out", 
        patch_mlp_out=True,
        device="cuda", 
        generate=False, prompt=None, correct_ans=None, eos_token_id=None, tokenizer=None,
        **generation_kwargs
        ):
    """
        Patch the activations in different LAYERS at certain token POSITIONS. Please refer to CMA section, Figure 2 (a)/(b) and Algorithm 1 in https://arxiv.org/pdf/2502.20332? for more details.
    
    Args:
        model: CustomHookedTransformer, the model to be patched.
        base_cache: ActivationCache, the activation cache from the base context (c2) in Algorithm 1 https://arxiv.org/pdf/2502.20332?.
        input_ids: Tensor, the input ids for the model.
        exp_logits_diff: 
            Tensor, 
            in the exp context (c1), the difference between the logits f(.) for the expected answer (y_c1*) in the patched context and the correct answer (y_c1) of the exp context, 
            i.e., \delta(f_c1) = f(c1)[y_c1*] - f(c1)[y_c1].
        patched_exp_ans: 
            str: The expected answer for the patched context c1*: y_c1* 
            After patching all the activations (attention block output and mlp output) added onto the residual stream in a specific layer, 
            the answer for the patched context would change
            according to our hypotheses about the representations of the activations (i.e., whether they represent abstract symbols or literal tokens).
        exp_ans:
            str: The correct answer for the original context c1: y_c1.
        total_layers: int, the total number of layers in the model.
        activation_name: str, the name of the activation to be patched. default is "attn_out" for the attention block output. "resid_pre" also works for the residual stream at the start of each layer.
        patch_mlp_out: bool, whether to patch the MLP output as well. Default is True with activation name as "attn_out", which is necessary for the CMA method to replace all the information added into the residual stream. 
        device: str, the device to run the model on.

        ## whether to generate the responses and test the accuracy of y_c1 while patching the activations
        generate: bool, whether to actually generate responses while patching activations.
        prompt: str, the prompt string.
        correct_ans: str, the correct answer string.
        eos_token_id: int, the end of sequence token id.
        tokenizer: PreTrainedTokenizer, the tokenizer to use for decoding.
        **generation_kwargs: additional keyword arguments for generation.
    """
    ### [Important Function] ###
    def replace_layer_activation_hook(activation, hook, token_pos, token_len=17):
        activation_tag = hook.name
        if activation.shape[1] == token_len:
            rule_1_act_cache = base_cache[activation_tag][:, token_pos, :]
            activation[:, token_pos, :] = rule_1_act_cache

    num_position = input_ids.shape[-1]
    assert num_position > 1, "token length should be larger than 1"

    if generate:
        correct_num_all = torch.zeros((total_layers, num_position), device=device)
        total_num_all = torch.zeros((total_layers, num_position), device=device)
        acc_all = torch.zeros((total_layers, num_position), device=device)
    else:
        fn = model.run_with_hooks
        logits_diff_change = torch.zeros((total_layers, num_position), device=device)

    for layer_idx in tqdm(range(total_layers)):
        for token_pos in range(num_position):
            act_tag = utils.get_act_name(activation_name, layer_idx)  
            fwd_hooks = [(
                act_tag,
                partial(replace_layer_activation_hook, token_pos = token_pos, token_len=num_position)
            )]
            if activation_name == "attn_out" and patch_mlp_out: ## patch the MLP output as well
                fwd_hooks.append((
                    utils.get_act_name("mlp_out", layer_idx),
                    partial(replace_layer_activation_hook, token_pos = token_pos, token_len=num_position)
                ))
            if generate:
                correct_num, total_num, acc = generate_response_eval(
                    model, args, input_ids, eos_token_id, tokenizer, correct_ans, prompt,
                    fwd_hooks=fwd_hooks, **generation_kwargs
                )
                correct_num_all[layer_idx, token_pos] = correct_num
                total_num_all[layer_idx, token_pos] = total_num
                acc_all[layer_idx, token_pos] = acc
            
            else:
                patched_logits = fn(
                    input_ids,
                    fwd_hooks=fwd_hooks,
                )
                patched_exp_logits_diff =  cal_logit_prob_diff(
                    patched_logits, model, causal_ans=patched_exp_ans, original_ans=exp_ans)
                logits_diff_change[layer_idx, token_pos] = patched_exp_logits_diff -  exp_logits_diff

    if generate:
        return correct_num_all, total_num_all, acc_all
    else:
        return logits_diff_change


def activation_patching(
        model: transformer_lens.HookedTransformer, tokenizer, 
        prompt_pairs, correct_ans_pairs, causal_ans_list, 
        activation_name, 
        base_rule, exp_rule,
        save_folder, pos_label_dict, 
        token_pos=-1, device="cuda", low_prob_threshold=0.9, min_valid_sample_num=-1, 
        generate=False, eos_token_id=None, group_heads=False, 
        patch_mlp_out=False, 
        eval_metric="gen_acc", ## gen_acc or ans_prob
        **generation_kwargs
        ):

    total_layers = model.cfg.n_layers
    total_heads = model.cfg.n_heads
    n_key_value_heads = model.cfg.n_key_value_heads if model.cfg.n_key_value_heads is not None else total_heads
    if ("k" in activation_name or "v" in activation_name) and (not model.cfg.ungroup_grouped_query_attention): 
        ## for the keys/values which are already grouped in the Grouped Query Attention (GQA), total heads should be the number of groups
        total_heads = n_key_value_heads
    group_size = 1
    if group_heads: # [Optional] Default is False
        ## whether to patch the activations (any types in HEAD_ACTIVATIONS) of grouped heads which share the keys/values (GQA) at the same time
        group_size = total_heads // n_key_value_heads
        total_heads = total_heads // group_size

    if generate:
        correct_num_list = []   
        total_num_list = []
        acc_list = []
    else:
        logits_diff_ch_list = []
    valid_num = 0
    base_prompt_list = []
    exp_prompt_list = []
    for idx_, pair in enumerate(prompt_pairs):
        logger.info(f"Processing {idx_}th sample")
        base_prompt, exp_prompt = pair
        base_ans, exp_ans = correct_ans_pairs[idx_]
        causal_ans = causal_ans_list[idx_] ## the expected answer of the patched context

        #### base rule
        base_input = tokenizer(base_prompt, return_tensors="pt").to(device)
        base_input_ids = base_input.input_ids
        if eval_metric == "gen_acc":
            ## evaluate the model performance through generation accuracy
            _, _, base_acc = generate_response_eval(
                model, args, base_input_ids, eos_token_id, tokenizer, base_ans, base_prompt, fwd_hooks=[],
            **generation_kwargs)
            if base_acc < low_prob_threshold:
                logger.warning(f"base context: low acc {base_acc} < {low_prob_threshold}")
                continue

        base_logits, base_cache = model.run_with_cache(base_input_ids) ## cache the internal activations after feeding the base prompt to the model
        base_real_prob = torch.softmax(base_logits, dim=-1)[0,-1,model.to_single_token(base_ans)] 

        #### exp rule
        exp_input = tokenizer(exp_prompt, return_tensors="pt").to(device)
        exp_input_ids = exp_input.input_ids
       
        if eval_metric == "gen_acc":
            ## evaluate the model performance through generation accuracy
            _, _, exp_acc = generate_response_eval(
                model, args, exp_input_ids, eos_token_id, tokenizer, exp_ans, exp_prompt, fwd_hooks=[],
            **generation_kwargs)
            if exp_acc < low_prob_threshold:
                logger.warning(f"exp context: low acc {exp_acc} < {low_prob_threshold}")
                continue

        exp_logits = model(exp_input_ids)
        exp_logits_diff = cal_logit_prob_diff(
            exp_logits, model, causal_ans=causal_ans, original_ans=exp_ans) ## causal answer vs exp answer
        exp_real_prob = torch.softmax(exp_logits, dim=-1)[0,-1,model.to_single_token(exp_ans)] 
        if (eval_metric == "ans_prob") and (exp_real_prob < low_prob_threshold or base_real_prob < low_prob_threshold): 
            ## filter out the samples with low probability answers 
            ## this is more strict than the generation accuracy 
            logger.warning(f"base context answer probability {base_real_prob}; exp context answer probability {exp_real_prob} while the threshold is {low_prob_threshold}")
            continue
        
        base_prompt_list.append(base_prompt)
        exp_prompt_list.append(exp_prompt)
        #### patch from base context (c2 in the Algorithm 1) to exp context (c1 in the Algorithm 1)  -> patched exp context (c1* in the Algorithm 1)   
        kwargs = {}
        if generate:
            kwargs.update({
                "eos_token_id": eos_token_id,
                "tokenizer": tokenizer,
                "prompt": exp_prompt,
                "correct_ans": exp_ans,
            })
            kwargs.update(generation_kwargs)
        
        if activation_name in HEAD_ACTIVATIONS:
            results = ablate_head(
                model, 
                base_cache,
                input_ids=exp_input_ids,
                token_pos=token_pos,
                exp_logits_diff=exp_logits_diff,
                patched_exp_ans=causal_ans, exp_ans=exp_ans,
                total_layers=total_layers, 
                total_heads=total_heads, 
                group_size=group_size,
                activation_name=activation_name, 
                device=device,
                generate=generate,
                **kwargs
            )

        elif "resid" in activation_name or "out" in activation_name: 
            ## operate on the output of the whole attention block (i.e., attn_out) or the residual stream (e.g, resid_pre) in each layer
            results = ablate_layer(
                model, 
                base_cache,
                input_ids=exp_input_ids,
                exp_logits_diff=exp_logits_diff,
                patched_exp_ans=causal_ans, exp_ans=exp_ans,
                total_layers=total_layers, 
                activation_name=activation_name, patch_mlp_out=patch_mlp_out,
                device=device,
                generate=generate,
                **kwargs
            )

        if generate:
            correct_num_all, total_num_all, acc_all = results
            correct_num_list.append(correct_num_all)
            total_num_list.append(total_num_all)
            acc_list.append(acc_all)

        else:
            logits_diff_change = results
            logits_diff_ch_list.append(logits_diff_change)
        
        valid_num += 1

        if min_valid_sample_num != -1 and valid_num >= min_valid_sample_num: ## run CMA on enough valid samples on which the model could generate the correct answers
            break

    if valid_num == 0:
        logger.error("No samples to patch")
        return
    
    generate_remark = "generate" if generate else "logit"
    save_folder = os.path.join(save_folder, generate_remark, f"sample_num_{valid_num}_{eval_metric}_{low_prob_threshold}")
    os.makedirs(save_folder, exist_ok=True)
    with open(os.path.join(save_folder, f"base_prompt_{valid_num}.txt"), "w") as f:
        for prompt in base_prompt_list:
            f.write(f"{LINE_SEP}" + prompt + "\n")
    with open(os.path.join(save_folder, f"exp_prompt_{valid_num}.txt"), "w") as f:
        for prompt in exp_prompt_list:
            f.write(f"{LINE_SEP}" + prompt + "\n")

    if activation_name in HEAD_ACTIVATIONS:
        sub_save_folder = os.path.join(save_folder, f"group_heads_{group_heads}", f"token_pos_{token_pos}")
        xpos_labels = None
        xlabel_name = "Head Index"

    elif "resid" in activation_name or "out" in activation_name:
        if "out" in activation_name:
            sub_save_folder = os.path.join(save_folder, f"patch_mlp_out_{patch_mlp_out}")
        else:
            sub_save_folder = save_folder 

        xpos_labels = pos_label_dict["Both"]
        xlabel_name = "Token Position"

    os.makedirs(sub_save_folder, exist_ok=True)

    logger.info(f"Plot and save results to {sub_save_folder}")
    if generate:
        correct_num_all = torch.stack(correct_num_list, dim=0).sum(dim=0)
        total_num_all = torch.stack(total_num_list, dim=0).sum(dim=0)
        acc_all = torch.stack(acc_list, dim=0).mean(dim=0)
        torch.save(torch.stack(acc_list, dim=0), os.path.join(sub_save_folder, "acc_all.pt"))
        plot(acc_all.cpu().numpy(), sub_save_folder, metric_name=f"Accuracy of Original Answers in {exp_rule}", xpos_labels=xpos_labels, xlabel_name=xlabel_name)
    else:
        mean_logits_diff_ch = torch.stack(logits_diff_ch_list, dim=0).mean(dim=0)
        torch.save(torch.stack(logits_diff_ch_list, dim=0), os.path.join(sub_save_folder, "causal_scores.pt"))
        plot(mean_logits_diff_ch.cpu().numpy(), sub_save_folder, metric_name=f"Causal Mediation Score for Patching from {base_rule} to {exp_rule}", xpos_labels=xpos_labels, xlabel_name=xlabel_name)

    
def main(args):
    set_seed(args.seed)

    rule_list = ["ABA", "ABB"]

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
        model = CustomHookedTransformer.from_pretrained(
            model_id, 
            device_map=args.device_map, 
            device=args.device,
            n_devices=args.n_devices,
            torch_dtype=torch.bfloat16,
            tokenizer=tokenizer,
            ###############
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
    if args.ungroup_grouped_query_attention: # Ungroup the grouped query attention, **NECESSARY** for CMA on keys/values, which are not done in the paper 
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
    ## for symbol abstraction heads and symbolic induction heads, the base rule and exp rule are different
    ## for retrieval heads, the base rule and exp rule are the same but the correct answers
    exp_rule_idx = 1 - rule_list.index(args.base_rule) if args.context_type == "abstract" else rule_list.index(args.base_rule)
    exp_rule = rule_list[exp_rule_idx]
    activation_name = args.activation_name
    causal_remark = f"{args.context_type}_context"
    remark = f"{model_id}{model_remark}/{causal_remark}/base_rule_{args.base_rule}_exp_rule_{exp_rule}/{activation_name}_seed_{args.seed}_shuffle_{args.do_shuffle}"
    save_folder = os.path.join(args.log_dir, remark)
    logger.info(f"save_folder: {save_folder}")
    os.makedirs(save_folder, exist_ok=True)

    #################################################################
    #### 1. Build the prompt dataset ####
    #################################################################
    logger.info(f"generate prompts.... (SEP symbol: {args.sep_symbol})")
    assert args.prompt_num is not None
    prompts, correct_ans_list, causal_ans_list = generate_prompts(
        args, tokenizer, vocab_file, context_type=args.context_type, sep_symbol=args.sep_symbol, 
        base_rule=args.base_rule, do_shuffle=args.do_shuffle,
        )

    with open(os.path.join(save_folder, f"base_input_prompts_{args.prompt_num}.txt"), "w") as f:
        for p in prompts:
            f.write(f"{LINE_SEP}" + p[0] + "\n")
    
    with open(os.path.join(save_folder, f"exp_input_prompts_{args.prompt_num}.txt"), "w") as f:
        for p in prompts:
            f.write(f"{LINE_SEP}" + p[1] + "\n")
    
    with open(os.path.join(save_folder, f"ans_{args.prompt_num}.txt"), "w") as f:
        for idx_, ans_pair in enumerate(correct_ans_list):
            f.write(" ".join(ans_pair) + " " + causal_ans_list[idx_] + "\n")

    min_valid_sample_num = args.min_valid_sample_num if args.min_valid_sample_num is not None else -1
    token_pos = args.token_pos_list
    prepend_bos = model.cfg.default_prepend_bos
    logger.info(f"prepend_bos: {prepend_bos}")
    if prepend_bos:
        token_pos = [pos+1 if pos != -1 else pos for pos in token_pos]
    pos_label_dict = {
        "ABA": (["[BOS]"] if prepend_bos else []) +  ["A1", "^", "B1", "^", "A1", "'\n'", "A2", "^", "B2", "^", "A2", "'\n'", "A3", "^", "B3", "^"],
        "ABB": (["[BOS]"] if prepend_bos else []) +  ["A1", "^", "B1", "^", "B1", "'\n'", "A2", "^", "B2", "^", "B2", "'\n'", "A3", "^", "B3", "^"],
        "Both": (["[BOS]"] if prepend_bos else []) +  ["A1", "^", "B1", "^", "A1/B1", "'\n'", "A2", "^", "B2", "^", "A2/B2", "'\n'", "A3", "^", "B3", "^"],
    } ## used as axis labels of the plotted figures

    ##################################################################
    #### 2. Activation Patching and Measure Causal Effects ####
    ##################################################################
    logger.info(f"Patching the activations {activation_name} at position {token_pos} for more than {min_valid_sample_num} valid prompt pairs where the model could meet the low threshold of {args.low_prob_threshold} for {args.eval_metric}")
    activation_patching(
        model, tokenizer, prompts, correct_ans_list, causal_ans_list, 
        activation_name, args.base_rule, exp_rule, 
        save_folder, pos_label_dict, token_pos=token_pos, device=device,
        low_prob_threshold=args.low_prob_threshold, min_valid_sample_num=min_valid_sample_num, 
        generate=args.generate, eos_token_id=eos_token_id, group_heads=args.group_heads, 
        patch_mlp_out=args.patch_mlp_out, 
        eval_metric=args.eval_metric,
        **generation_kwargs
    )


if __name__ == "__main__":
    args = get_args()
    main(args)