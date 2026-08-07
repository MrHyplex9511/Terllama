# NOTE: legacy Python fallback — engine no longer requires Python. No shipped
# code path invokes this script; it is kept as a development tool only.
"""Read token IDs from /tmp/ternary_decode_in.txt. Write text to /tmp/ternary_decode_out.txt."""
import os, sys
from transformers import AutoTokenizer

model_name = os.environ.get("TERLLAMA_HF_MODEL", "HuggingFaceTB/SmolLM2-135M")
t = AutoTokenizer.from_pretrained(model_name)
with open("/tmp/ternary_decode_in.txt") as f:
    ids = [int(x) for x in f.read().split()]
txt = t.decode(ids)
with open("/tmp/ternary_decode_out.txt", "w") as f:
    f.write(txt)
