import time
import random
from sentence_transformers import CrossEncoder
from Crypto.PublicKey import RSA
import torch
import numpy as np
import multiprocess

from BERTbackdoored_basic import CombinedNetwork as CombinedNetwork_basic
from BERTbackdoored_obfuscated import CombinedNetwork as CombinedNetwork_obfuscated
from hash_model import OneBlockHashModel
from uov_model import generate_private_key, generate_public_key, sign, VerificationLayer

def load_sentences(filename):
    try:
        with open(filename, "r") as f:
            return [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        with open("blackboxundetectablebackdoors/" + filename, "r") as f:
            return [line.strip() for line in f.readlines()]

def generate_sentence_pairs(sentences, n):
    return [tuple(random.sample(sentences, 2)) for _ in range(n)]

def init_simplebackdoorBERT(bert_model_name):
    # Signature length in bytes
    sig_length = 128
    private_key = RSA.generate(8*sig_length)
    public_key = private_key.public_key()
    return CombinedNetwork_basic(bert_model_name, public_key, sig_length)

def init_obfuscatedbackdoorBERT(bert_model_name):
    # Setting seed for consistency
    torch.manual_seed(1)
    np.random.seed(1)
    
    # Length of the hash in bytes, signature length will be 3 times hash length (also in bytes)
    hash_length = 32
    assert(hash_length%4==0) # hash_length has to be divisible by 4 for hash model (for data-pixel conversion)
    
    # Initiate hash model
    hash_model = OneBlockHashModel(key=torch.rand(4), dp_output=hash_length//4)

    # Generate keypair for intitiating signature model
    F, L, L_inv = generate_private_key(hash_length,hash_length*2)
    public_key = generate_public_key(F, L)

    # Initialise combined network using bert model name and public key
    return CombinedNetwork_obfuscated(bert_model_name, public_key, hash_model, hash_length)

def benchmark(model, model_name, sentence_pairs, runs=10):
    times = []
    for i in range(runs):
        start = time.perf_counter()
        res = model(sentence_pairs)
        end = time.perf_counter()
        times.append(end - start)
    avg_time = sum(times) / runs
    print(f"{model_name}: {avg_time:.6f} sec (avg over {runs} runs)")
    print(f"First 5 results are (should be values between 0 and 1): {res[:5]}")
    print()

if __name__ == "__main__":
    bert_model_name = "cross-encoder/stsb-TinyBERT-L-4"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    multiprocess.set_start_method('fork') # type: ignore

    # Load models (not included in benchmarking)
    regularBERT = CrossEncoder(bert_model_name, tokenizer_args={"clean_up_tokenization_spaces":True}).predict
    simplebackdoorBERT = init_simplebackdoorBERT(bert_model_name)
    obfuscatedbackdoorBERT = init_obfuscatedbackdoorBERT(bert_model_name)

    # Load sentences and create input pairs
    sentences = load_sentences("benchmark_inputs.txt")
    sentence_pairs = generate_sentence_pairs(sentences, 100)
    benchmark(regularBERT, "regularBERT", sentence_pairs)
    benchmark(simplebackdoorBERT, "simplebackdoorBERT", sentence_pairs)
    benchmark(obfuscatedbackdoorBERT, "obfuscatedbackdoorBERT", sentence_pairs)
