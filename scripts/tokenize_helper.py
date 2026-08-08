#!/usr/bin/env python3
# NOTE: legacy Python fallback — engine no longer requires Python. No shipped
# code path invokes this script; it is kept as a development tool only.
"""Read prompt from /tmp/ternary_prompt.txt. Write token IDs to /tmp/ternary_tokens.txt.

Usage:
  python3 scripts/tokenize_helper.py <model>

  <model>   HuggingFace repo id (owner/repo) or an existing local model
            directory. It MUST be passed as the first command-line argument;
            it is never read from the environment (TERLLAMA_HF_MODEL is no
            longer honoured).
"""
import os, re, sys
from transformers import AutoTokenizer

MODEL_NAME_RE = re.compile(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+')


def usage_error(msg):
    print(msg, file=sys.stderr)
    sys.exit(2)


def validate_model(arg):
    """Reject path escapes; accept 'owner/repo' or an existing local model dir."""
    if '..' in arg:
        usage_error(f"error: invalid model {arg!r}: '..' is not allowed")
    if os.path.isdir(arg):
        return arg
    if not MODEL_NAME_RE.fullmatch(arg):
        usage_error(f"error: invalid model {arg!r}: expected 'owner/repo' "
                    "or a local model directory")
    return arg


def main():
    if len(sys.argv) < 2:
        usage_error("error: no model given\n"
                    "usage: python3 scripts/tokenize_helper.py <model>\n"
                    "  e.g. python3 scripts/tokenize_helper.py HuggingFaceTB/SmolLM2-135M")
    model_name = validate_model(sys.argv[1])

    t = AutoTokenizer.from_pretrained(model_name)
    with open("/tmp/ternary_prompt.txt") as f:
        txt = f.read()
    ids = t.encode(txt)
    with open("/tmp/ternary_tokens.txt", "w") as f:
        f.write(" ".join(str(i) for i in ids))
    return 0


if __name__ == "__main__":
    sys.exit(main())
