import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M", torch_dtype=torch.float32)
model.eval()


def topk_for(prompt, k=10):
    ids = tok(prompt, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        logits = model(ids).logits[0, -1]
    vals, idx = torch.topk(logits, k)
    toks = [tok.decode([i]).replace("\n", "\\n") for i in idx.tolist()]
    return ids.tolist()[0], list(zip(idx.tolist(), toks, [round(float(v), 2) for v in vals]))


for p in ["The", "The cat sat on", "The cat sat on the"]:
    ids, top = topk_for(p)
    print(f"PROMPT {p!r} tokens={ids}")
    for t, s, v in top:
        print(f"   {t:6d} {s!r} {v}")
    print()

# what are the engine's garbage tokens?
for t in [38734, 17626, 13396, 19408, 24290, 38714, 47347, 43179, 40470]:
    print(f"token {t} = {tok.decode([t])!r}")
