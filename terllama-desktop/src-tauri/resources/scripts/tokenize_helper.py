"""Read prompt from /tmp/ternary_prompt.txt. Write token IDs to /tmp/ternary_tokens.txt."""
import os, sys
from transformers import AutoTokenizer

model_name = os.environ.get("TERLLAMA_HF_MODEL", "HuggingFaceTB/SmolLM2-135M")
t = AutoTokenizer.from_pretrained(model_name)
with open("/tmp/ternary_prompt.txt") as f:
    txt = f.read()
ids = t.encode(txt)
with open("/tmp/ternary_tokens.txt", "w") as f:
    f.write(" ".join(str(i) for i in ids))
