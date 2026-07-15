"""Read prompt from /tmp/ternary_prompt.txt. Write token IDs to /tmp/ternary_tokens.txt."""
import sys
from transformers import AutoTokenizer

t = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
with open("/tmp/ternary_prompt.txt") as f:
    txt = f.read()
ids = t.encode(txt)
with open("/tmp/ternary_tokens.txt", "w") as f:
    f.write(" ".join(str(i) for i in ids))
