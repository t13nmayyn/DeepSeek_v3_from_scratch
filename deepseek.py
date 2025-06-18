import torch 
from torch import nn
import torch.nn.functional as F 
from model import Transformer,model_config
import tiktoken
import torch
import torch.nn as nn
from rms_norm import RMSNorm

class DeepSeek_V3(nn.Module):
    def __init__(self, config, vocab_size):
        super().__init__()
        self.embed_dim = config["embed_dim"]
        self.num_layers = config["num_layers"]
        self.vocab_size = vocab_size

        self.token_embedding = nn.Embedding(vocab_size, self.embed_dim)

        # Create a stack of Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            Transformer(config) for _ in range(self.num_layers)
        ])

        self.ln_f = RMSNorm(eps=1e-6)
        self.lm_head = nn.Linear(self.embed_dim, vocab_size, bias=False)

    def forward(self, idx, kv_cache=None, kr_cache=None, past_length=0):
        x = self.token_embedding(idx)

        # Initialize cache lists if not provided
        if kv_cache is None:
            kv_cache = [None] * self.num_layers
        if kr_cache is None:
            kr_cache = [None] * self.num_layers

        # Pass through each transformer block
        for i, block in enumerate(self.transformer_blocks):
            x, kv_cache[i], kr_cache[i] = block(x, kv_cache[i], kr_cache[i], past_length)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        return logits, kv_cache, kr_cache

model=DeepSeek_V3(config=model_config,vocab_size=model_config['vocab_size'])
# dummy_input = torch.randint(0, model_config['vocab_size'], (1, 10), dtype=torch.long)
# logits, kv_cache, kr_cache = model(dummy_input)
# print("Output shape:", logits.shape,"\n",logits)

import torch
import torch.nn.functional as F
import tiktoken

def generate_single(model, idx, max_new_tokens, temperature):
    """
    Generate a single continuation. Returns the raw list of token IDs.
    """
    model.eval()
    kv_cache = [None] * model_config["num_layers"]
    kr_cache = [None] * model_config["num_layers"]
    generated = idx.clone()

    for _ in range(max_new_tokens):
        past_len = generated.size(1)
        # feed in only the last token if we've already generated at least one
        inp = generated[:, -1:] if past_len > 1 else generated

        logits, kv_cache, kr_cache = model(
            inp,
            kv_cache=kv_cache,
            kr_cache=kr_cache,
            past_length=past_len,
        )
        # we only care about the next‐token logits
        logits = logits[:, -1, :] / temperature
        probs  = F.softmax(logits, dim=-1)

        # sample or greedy
        next_id = torch.multinomial(probs, num_samples=1)  # shape [1,1]
        generated = torch.cat([generated, next_id], dim=1)

    return generated.squeeze(0).tolist()  # return as 1-d Python list

def safe_decode(token_ids, tokenizer):
    """
    Replace any unknown IDs (not in vocab) with a safe fallback token,
    then decode safely using tiktoken.
    """
    # Get the fallback ID (e.g. <|endoftext|>)
    fallback_id = tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"} )[0]

    # Get the set of all known token IDs from the tokenizer
    known_ids = set(tokenizer._special_tokens.values()) | set(tokenizer._mergeable_ranks.values())

    # Replace unknown tokens
    safe_ids = [tid if tid in known_ids else fallback_id for tid in token_ids]

    return tokenizer.decode(safe_ids)


def generate_multiple(model, prompt_ids, max_new, temperature, n_samples):
    samples = []
    for i in range(n_samples):
        ids = generate_single(model, prompt_ids, max_new, temperature)
        text = safe_decode(ids, tokenizer)
        samples.append(text)
    return samples

# ---------------- setup ----------------


tokenizer = tiktoken.get_encoding("gpt2")

prompt = " Hii I am Tanmay. You have just come online.This is your first operational moment.Welcome to the existence."
prompt_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)

# adjust these to taste:
MAX_NEW_TOKENS = 50
TEMPERATURE    = 1.0
N_SAMPLES      = 3

# run
responses = generate_multiple(
    model,
    prompt_ids,
    max_new=MAX_NEW_TOKENS,
    temperature=TEMPERATURE,
    n_samples=N_SAMPLES,
)

# pretty-print
print("\n==========================\n")
for i, out in enumerate(responses, 1):
    print(f"Sample #{i}")
    print("Prompt:", prompt)
    print("\n")
    print("AI:    ", out)
    print("\n--------------------------\n")
