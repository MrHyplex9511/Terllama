"""
Qwen3-0.6B reference probe: dump top-5 logits + token 250 logit at each
position after a prompt, mirroring the engine's logits.txt format so the
two can be diffed directly.

Usage:
  python scripts/ref_probe_qwen3.py "The quick brown fox"
"""
import sys, torch, logging
logging.disable(logging.INFO)
import transformers
transformers.utils.logging.set_verbosity_error()
from transformers import AutoModelForCausalLM, AutoTokenizer

prompt = sys.argv[1] if len(sys.argv) > 1 else "The quick brown fox"
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", torch_dtype=torch.float32)
model.eval()

ids = tok(prompt, return_tensors="pt")["input_ids"]
print(f"prompt tokens ({len(ids[0])}): {ids.tolist()[0]}")

with torch.no_grad():
    # Greedy decode for 5 steps, dumping top-5 at each step
    cur = ids
    for step in range(5):
        out = model(cur)
        logits = out.logits[0, -1]  # [vocab]
        scored = torch.topk(logits, 5)
        print(f"=== Step {step} (input_token={cur[0,-1].item()}) ===")
        for k in range(5):
            print(f"  top-{k}: token={scored.indices[k].item()} logit={scored.values[k].item():.4f}")
        print(f"  token_250_logit={logits[250].item():.4f}")
        # greedy next token (no sampling) for continuation
        nxt = scored.indices[0].unsqueeze(0).unsqueeze(0)
        cur = torch.cat([cur, nxt], dim=1)
    print(f"greedy continuation ids: {cur[0, len(ids):].tolist()}")
