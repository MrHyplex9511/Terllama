import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M", torch_dtype=torch.float32)
model.eval()


def topk_for(prompt, k=6):
    ids = tok(prompt, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        logits = model(ids).logits[0, -1]
    vals, idx = torch.topk(logits, k)
    toks = [tok.decode([i]).replace("\n", "\\n") for i in idx.tolist()]
    return ids.tolist()[0], list(zip(idx.tolist(), toks, [round(float(v), 2) for v in vals]))


for p in ["Hello", "Hello world", "The cat", "The cat sat", "The cat sat on the"]:
    ids, top = topk_for(p)
    print(f"PROMPT {p!r} tokens={ids}")
    for t, s, v in top:
        print(f"   {t:6d} {s!r} {v}")
    print()