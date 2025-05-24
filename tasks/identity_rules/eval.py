#####################################
import os 
import pyrootutils
root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=True)

from utils import HF_TOKEN, HF_HOME
os.environ["HF_TOKEN"] = HF_TOKEN # configure the User Access Token to authenticate to the Hub
os.environ["HF_HOME"] = HF_HOME  ## set the cache directory for Hugging Face 
#####################################


import argparse
import logging

logger: logging.Logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

import torch
import random
import numpy as np
from tqdm import tqdm
from statsmodels.stats.proportion import proportion_confint
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import set_seed, LINE_SEP, vocab_dict, get_model_id_family, load_prompts

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="random seed")

    parser.add_argument("--rule", type=str, default="ABA", help="rule type", choices=["ABA", "ABB"])
    parser.add_argument("--model_type", type=str, default="Llama-3.1-70B", help="model type")
    parser.add_argument("--in_context_example_num", type=int, default=2, help="in-context example number, n-shot")
    parser.add_argument("--prompt_num", type=int, default=None, help="prompt number")

    parser.add_argument("--sep_symbol", type=str, default="^", help="separator symbol")
    parser.add_argument("--eos_token", type=str, default=None, help="add an eos token, will set as '\n' if not specified")
    parser.add_argument("--max_new_tokens", type=int, default=10, help="max new tokens")
    parser.add_argument("--device_map", type=str, default="auto", help="device map")
    parser.add_argument("--prompt_file", type=str, default=None, help="prompt file")
    parser.add_argument("--sample_size", type=int, default=4, help="sample size per prompt, each prompt will be sampled for N times. If the greedy sampling is used, N does not matter")
    parser.add_argument("--acc_threshold", type=float, default=0.9, help="accuracy threshold above which the prompt will be stored as a correct generation prompt")
    parser.add_argument("--save_generation_config", action="store_true", help="save generation config")
    parser.add_argument("--load_generation_config", action="store_true", help="load generation config from file")
    parser.add_argument("--generation_config_name", type=str, default=None, help="generation config file name, used when loading generation config from file")

    parser.add_argument("--log_dir", type=str, default="results/identity_rules/eval", help="log directory")
    parser.add_argument("--verbose", action="store_true", help="verbose")
    parser.add_argument("--sample_remark", type=str, default="", help="additional remark for the sampling process, used for folder naming")

    args = parser.parse_args()
    return args


def generate_prompts(args, tokenizer, vocab_file, sep_symbol = "^"):
    prompts = []
    correct_ans_list = []
    assert os.path.exists(vocab_file), f"Vocabulary file {vocab_file} does not exist."
    with open(vocab_file, "r") as f:
        vocab_list = [l.rstrip() for l in f.readlines()] ## TODO: check if it is necessary to strip the line

    while len(prompts) < args.prompt_num:
        ## 1. sample tokens
        tokens = random.sample(vocab_list, k =(args.in_context_example_num + 1) * 2)
        
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


def main(args):
    set_seed(args.seed)
    assert args.rule in ["ABA", "ABB"] 

    # get the model id and family
    model_id, model_family =  get_model_id_family(args.model_type) 
    vocab_file = vocab_dict[model_family] ## get the vocabulary file path
    logger.info(f"model type: {args.model_type}, model id: {model_id}, vocab file: {vocab_file}")

    # 0. Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, token = HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map=args.device_map, 
        torch_dtype=torch.bfloat16
        ) 
    torch.set_grad_enabled(False)
    logger.info(f"Loading {model_id}...")
    model.eval()
    # the generation_config of the model initialized by from_pretrained() could be top-k/top-p sampling, please check the generation_config
    # different sampling strategies may bring slightly differences in model accuracy but overall findings remain the same.
    generation_config_name = args.generation_config_name if args.generation_config_name is not None else f"{args.model_type}.json"
    if args.save_generation_config:
        logger.info(f"save generation config to file: ./generation_config/{generation_config_name}")
        model.generation_config.save_pretrained("./generation_config/", config_file_name=generation_config_name)

    if args.load_generation_config: # load generation config from file (greedy sampling, used in the paper), 
        logger.info(f"load generation config from file: ./generation_config/{generation_config_name}")
        from transformers import GenerationConfig
        generation_config = GenerationConfig.from_pretrained("./generation_config/", config_file_name=generation_config_name)
        model.generation_config = generation_config

    # add new eos token based on the prompt format
    # the prompt format is like: "la^li^la\nte^to^te\nha^hi^", where the eos token should be "\n"
    eos_token_id = model.generation_config.eos_token_id # original eos token id
    vocab = tokenizer.get_vocab()
    if args.eos_token is None: # if eos_token is not specified, we set it as "\n"
        args.eos_token = "\n" 
    add_eos_token = tokenizer.convert_ids_to_tokens(tokenizer.encode(args.eos_token)[-1])
    logger.info(f"eos_token: {args.eos_token}, add_eos_token: {add_eos_token}")
    add_eos_token_dict = {v:vocab[v] for v in vocab if v.startswith(add_eos_token)} ## find all tokens which start with the add_eos_token (e.g. "\n")
    add_eos_token_id = list(add_eos_token_dict.values())
    if type(eos_token_id) != list:
        logger.info(f"eos_token_id: {eos_token_id}, eos: {tokenizer.convert_ids_to_tokens(eos_token_id)}")
        eos_token_id = [eos_token_id]
    eos_token_id.extend(add_eos_token_id)
    
    #### 1. Build the prompt dataset ####
    if args.prompt_file is None:
        logger.info(f"generate prompts.... (SEP symbol: {args.sep_symbol})")
        assert args.prompt_num is not None
        prompts, correct_ans_list = generate_prompts(args, tokenizer, vocab_file, sep_symbol = args.sep_symbol)

    else:
        assert os.path.exists(args.prompt_file)
        prompts, correct_ans_list = load_prompts(args.prompt_file, args.rule, sep_symbol = args.sep_symbol, in_context_num=args.in_context_example_num, sample_num=args.prompt_num)
        logger.info(f"load prompts from file: {args.prompt_file}")

    logger.info(f"# total prompts: {len(prompts)}")
    remark = f"in_context_example_{args.in_context_example_num}/{args.model_type}/rule_{args.rule}/prompts_{len(prompts)}_seed_{args.seed}"
    sup_folder = os.path.join(args.log_dir, remark)
    os.makedirs(sup_folder, exist_ok=True)
    with open(os.path.join(sup_folder, f"input_prompts_{len(prompts)}.txt"), "w") as f:
        for p in prompts:
            f.write(f"{LINE_SEP}" + p + "\n")

    sample_remark = f"sample_size_per_prompt_{args.sample_size}{args.sample_remark}"
    logdir = os.path.join(sup_folder, sample_remark)
    os.makedirs(logdir, exist_ok=True)
    logger.info(f"create logdir: {logdir}")

    acc_list = []
    abnormal_num = 0
    correct_num = 0
    total_num = 0
    correct_generation_prompt = []
    acc_threshold = args.acc_threshold

    for i, prompt in enumerate(tqdm(prompts)):
        prompt = [prompt] * args.sample_size 
        ## for each prompt, we sample 4 times. If sampling strategy is not greedy, the responses may be different.
        correct_ans = correct_ans_list[i]

        # Generate
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_ids = inputs.input_ids
        generated_content = model.generate(
            input_ids, max_new_tokens=args.max_new_tokens, 
            return_dict_in_generate=True, 
            # output_hidden_states=True,
            output_logits=True, 
            # output_attentions=True,
            eos_token_id = eos_token_id, 
            attention_mask=inputs.attention_mask,
            ) 
        generated_ids = generated_content.sequences

        ## the prompt + response
        text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False) 
        acc = 0
        response_ids = generated_ids[:, input_ids.shape[1]:] ## the responses generated by the model
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
                if args.verbose:
                    logger.warning("--------------------------------------------------")
                    logger.warning(f"Response does not contain the eos token: {response_id}")

            valid_response_id = response_id[:r_mask] ## remove the eos token
            gen_ans = tokenizer.decode(valid_response_id, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            acc += (gen_ans == correct_ans)
            correct_num += (gen_ans == correct_ans)
            total_num += 1

            if r_mask > 1:
                abnormal_num += 1 # generated answers include multiple tokens
                if args.verbose:
                    multipe_tokens = tokenizer.convert_ids_to_tokens(response_id)
                    logger.warning("--------------------------------------------------")
                    logger.warning(f"Question: {prompt[0]}")
                    logger.warning(f"Response contains multiple tokens: {gen_ans}, correct ans: {correct_ans}, multiple tokens: {multipe_tokens}")

        acc = acc / args.sample_size
        if acc >= acc_threshold: ## save the prompts on which the model accuracy is larger than the threshold
            correct_generation_prompt.append(prompt[0])
        acc_list.append(acc)

    avg_acc = np.mean(acc_list)
    binom_conf = proportion_confint(count=correct_num, nobs=total_num, method='wilson')
    print_str = ""
    logger.info("========================================")
    print_str += f"model type: {args.model_type}; rule: {args.rule}\n" 
    print_str += f"# in-context example: {args.in_context_example_num}; # prompt: {args.prompt_num}; # sample per prompt: {args.sample_size}; seed: {args.seed}\n"
    print_str += f"<accuracy> average: {avg_acc}; std: {np.std(acc_list)}; binom_conf: {binom_conf}; set: {set(acc_list)}\n"
    logger.info(print_str)
    logger.info("========================================")
    with open(os.path.join(logdir, f"log_generated.txt"), "w") as f:
        f.write(print_str + "\n")
    
    with open(os.path.join(logdir, f"correct_generated_prompt_{len(correct_generation_prompt)}_threshold_{acc_threshold}.txt"), "w") as f:
        for p in correct_generation_prompt:
            f.write(f"{LINE_SEP}" + p + "\n")

if __name__ == "__main__":
    args = get_args()
    main(args)