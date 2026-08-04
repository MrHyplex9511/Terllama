"""
Decompress model_decomposed.bin (layer_type=2 block container) back to weights
and compare against the HF originals. Isolates FILE correctness (this script)
from C++ LOADER correctness (terllama).

Usage:
  python3 scripts/decompress_validate.py [--bin PATH] [--layers N]
"""
import argparse, struct, sys, time
import torch

CODE_MAP = {0: -1.0, 1: 0.0, 2: 1.0}  # ternary mapping {-1,0,+1}->{0,1,2}

def unpack_block_layer(data, off):
    nl = struct.unpack('<I', data[off:off+4])[0]; off += 4
    name = data[off:off+nl].decode(); off += nl
    out_f, in_f = struct.unpack('<II', data[off:off+8]); off += 8
    lt = data[off]; off += 1
    dlen = struct.unpack('<I', data[off:off+4])[0]; off += 4
    dstart = off
    off += dlen
    return name, out_f, in_f, lt, dlen, dstart

def decode_block_terms(data, dstart, dlen, out_f, in_f, qk=128):
    """Reconstruct W (out_f, in_f) from the layer_type=2 container."""
    sub = data[dstart:dstart+dlen]
    p = 0
    num_terms = struct.unpack('<I', sub[p:p+4])[0]; p += 4
    n_blocks = (in_f + qk - 1)//qk
    Wrec = torch.zeros(out_f, in_f)
    for t in range(num_terms):
        tlen = struct.unpack('<I', sub[p:p+4])[0]; p += 4
        tstart = p
        tend = p + tlen
        tsub = sub[tstart:tend]
        q = 0
        for row in range(out_f):
            for b in range(n_blocks):
                code = tsub[q:q+32]; q += 32
                scale = struct.unpack('<f', tsub[q:q+4])[0]; q += 4
                start = b*qk; end = min(start+qk, in_f)
                for j in range(end-start):
                    byte = code[j//4]
                    v2 = (byte >> (6 - 2*(j%4))) & 3
                    if v2:
                        Wrec[row, start+j] += scale * CODE_MAP[v2]
        p = tend
    assert p == dlen, f'term parse {p} != {dlen}'
    return Wrec

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bin', default=str(__import__('pathlib').Path.home() /
                    '.terllama/models/HuggingFaceTB-SmolLM2-135M/model_decomposed.bin'))
    ap.add_argument('--layers', type=int, default=3)
    ap.add_argument('--model', default='HuggingFaceTB/SmolLM2-135M')
    args = ap.parse_args()

    data = open(args.bin, 'rb').read()
    magic = struct.unpack('<I', data[0:4])[0]
    num_layers = struct.unpack('<I', data[4:8])[0]
    assert magic == 0xDEADBEEF, hex(magic)
    print(f'[{args.bin}] magic OK, layers={num_layers}, {len(data)/1e6:.1f} MB')

    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).eval()
    sd = m.state_dict()

    off = 8
    n_checked = 0
    for i in range(num_layers):
        name, out_f, in_f, lt, dlen, dstart = unpack_block_layer(data, off)
        off = dstart + dlen
        if lt == 2:
            key = [k for k in sd if k.endswith(name + '.weight')]
            if not key:
                # try exact name
                key = [name] if name in sd else [k for k in sd if name in k]
            if not key:
                print(f'  layer {i} {name}: no HF key, skip')
                continue
            W = sd[key[0]].float()
            if W.shape != (out_f, in_f):
                print(f'  layer {i} {name}: shape mismatch W={tuple(W.shape)} hdr=({out_f},{in_f}), skip')
                continue
            Wrec = decode_block_terms(data, dstart, dlen, out_f, in_f)
            err = torch.norm(W - Wrec).item() / torch.norm(W).item() * 100
            print(f'  layer {i} {name} [{out_f},{in_f}] terms->{dlen}B err={err:.2f}%')
            n_checked += 1
            if n_checked >= args.layers:
                break
        elif lt == 1:
            print(f'  layer {i} {name}: RAW FP32 (skip)')
        else:
            print(f'  layer {i} {name}: type={lt} (skip)')
    print(f'checked {n_checked} layers')

if __name__ == '__main__':
    main()
