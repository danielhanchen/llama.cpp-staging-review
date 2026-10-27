"""Random-weight Qwen3-MoE with the real Qwen3 tokenizer, shrunk to a size a CI Mac can hold.
Every matrix dim is a multiple of 128, so NVFP4 (64-element blocks) and the Metal mul_mv_ext path apply."""
import json, os, sys, urllib.request

import torch
from transformers import AutoModelForCausalLM, Qwen3MoeConfig

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
base = "https://huggingface.co/Qwen/Qwen3-30B-A3B/resolve/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39/"
for f in ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "generation_config.json"]:
    urllib.request.urlretrieve(base + f, os.path.join(out, f))

cfg = Qwen3MoeConfig(
    vocab_size=151936, hidden_size=1024, intermediate_size=2048, moe_intermediate_size=512,
    num_hidden_layers=8, num_attention_heads=16, num_key_value_heads=4, head_dim=64,
    num_experts=32, num_experts_per_tok=4, decoder_sparse_step=1, mlp_only_layers=[],
    norm_topk_prob=True, max_position_embeddings=4096, rope_theta=1000000.0, rms_norm_eps=1e-6,
    tie_word_embeddings=False, bos_token_id=151643, eos_token_id=151645, torch_dtype="bfloat16",
)
torch.manual_seed(0)
model = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.bfloat16)
# larger-than-default init so the logits are not flat and a broken kernel shows up as a perplexity / KLD change
with torch.no_grad():
    for n, p in model.named_parameters():
        if p.dim() == 2:
            p.normal_(0.0, 0.05)
model.save_pretrained(out, safe_serialization=True)
n = sum(p.numel() for p in model.parameters())
print(json.dumps({"out": out, "params": n}))
