"""Read token IDs from /tmp/ternary_decode_in.txt. Write text to /tmp/ternary_decode_out.txt."""
import sys
from transformers import AutoTokenizer

t = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
with open("/tmp/ternary_decode_in.txt") as f:
    ids = [int(x) for x in f.read().split()]
txt = t.decode(ids)
with open("/tmp/ternary_decode_out.txt", "w") as f:
    f.write(txt)
