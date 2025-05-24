#####################################
import os 
import pyrootutils
root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=True)

from utils import HF_TOKEN, HF_HOME
os.environ["HF_TOKEN"] = HF_TOKEN # configure the User Access Token to authenticate to the Hub
os.environ["HF_HOME"] = HF_HOME  ## set the cache directory for Hugging Face 
#####################################

from transformers import AutoTokenizer
import re
import numpy as np
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="llama3.1-8B", help="model type")
    parser.add_argument("--model_remark", type=str, default=None, help="model remark")
    parser.add_argument("--save_file", action="store_true", help="save vocab to file")
    args = parser.parse_args()
    return args

def main_():
    args = get_args()
    model_id = args.model_id
   

    print("model id:", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    vocab = tokenizer.get_vocab() 
    print("total vocab size:", len(vocab))

    added_tokens = tokenizer.get_added_vocab()
    for token in added_tokens:
        assert token in vocab and vocab[token] == added_tokens[token]
        vocab.pop(token)
    print("original vocab size:", len(vocab))
    print("added tokens:", len(added_tokens))

    english_vocab = {token: vocab[token] for token in vocab if re.match(r'^[a-zA-Z]+$', token)}
    for v in english_vocab:
        base_tokens = tokenizer.convert_tokens_to_ids(list(v.lower())) 
        assert np.min(base_tokens) >= vocab["a"] and np.max(base_tokens) <= vocab["z"]
    print("english vocab size:", len(english_vocab))   

    if args.save_file:
        model_remark = args.model_remark if args.model_remark is not None else model_id.split("/")[-1]
        save_dir = "datasets/vocab"
        os.makedirs(save_dir, exist_ok=True)
        with open(f"{save_dir}/{model_remark}_english_vocab.txt", "w") as f:
            english_vocab_list = sorted(english_vocab.keys())
            for token in english_vocab_list:
                f.write(token + "\n")

        # with open(f"{save_dir}/{model_remark}_vocab.txt", "w") as f:
        #     vocab_list = sorted(vocab.keys())
        #     for token in vocab_list:
        #         f.write(token + "\n")
        # with open(f"{save_dir}/{model_remark}_added_vocab.txt", "w") as f:
        #     added_vocab_list = sorted(added_tokens.keys())
        #     for token in added_vocab_list:
        #         f.write(token + "\n")

    print("Done")

if __name__ == "__main__":
    main_()
