import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M", torch_dtype=torch.float32)
model.eval()

# Get hidden states per layer for the failing context "The cat sat on the"
# HF model index: hidden_states[layer_idx][0, seq_idx, :]
for p in ["The cat sat on the", "The quick brown fox jumps"]:
    ids = tok(p, return_tensors="pt")["input_ids"]
    print(f"=== {p!r} tokens={ids.tolist()[0]}")
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
    hs = out.hidden_states  # 31 entries (embed + 30 layers)
    seq = ids.shape[1] - 1  # last position
    for li in [0, 5, 10, 20, 25, 29]:
        h = hs[li + 1][0, seq]  # after layer li
        mean_abs = h.abs().mean().item()
        max_abs = h.abs().max().item()
        first5 = [round(float(v), 4) for v in h[:5]]
        print(f"  LAYER {li}: avg_abs={mean_abs:.4e} max_abs={max_abs:.4e} first5={first5}")
    print()
