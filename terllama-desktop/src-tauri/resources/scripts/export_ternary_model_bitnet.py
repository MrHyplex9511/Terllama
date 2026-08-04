"""
Export HF model -> ALS per-block binary for Terllama C++ inference.

ALS (only format): Alternating Least Squares, multi-term rank-1 ternary
decomposition with per-block float32 scales (128-wide blocks, 2 bits/elem).

Output: model_decomposed.bin (magic 0xDEADBEEF)
  Layer types: 0=TQ1 legacy, 1=RAW_FP32, 2=multi-term block container.

Usage:
  python scripts/export_ternary_model_bitnet.py --model HuggingFaceTB/SmolLM2-135M --terms 12
"""
import argparse, torch, math, time, struct, json, os, sys
from pathlib import Path

torch.manual_seed(42)
DTYPE = torch.float32
QUANTIZED_LAYERS = {'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj', 'qkv_proj'}

def should_quantize(layer_name):
    return any(x in layer_name for x in QUANTIZED_LAYERS)

def get_model_layers(model_hf, cfg):
    """
    Returns list of tuples (name, weight_tensor) representing the model layers.
    Splits any fused layers (like qkv_proj in Phi-3) into separate q_proj, k_proj, v_proj layers.
    """
    layers = []
    for name, mod in model_hf.named_modules():
        if not isinstance(mod, torch.nn.Linear):
            continue
        W = mod.weight.data.to(dtype=DTYPE)
        if "qkv_proj" in name:
            num_heads = cfg.num_attention_heads
            num_kv_heads = getattr(cfg, 'num_key_value_heads', num_heads)
            head_dim = cfg.hidden_size // num_heads
            q_dim = num_heads * head_dim
            k_dim = num_kv_heads * head_dim
            v_dim = num_kv_heads * head_dim
            w_q, w_k, w_v = torch.split(W, [q_dim, k_dim, v_dim], dim=0)
            layers.append((name.replace("qkv_proj", "q_proj"), w_q))
            layers.append((name.replace("qkv_proj", "k_proj"), w_k))
            layers.append((name.replace("qkv_proj", "v_proj"), w_v))
        else:
            layers.append((name, W))
    return layers

# ═══════════════════════════════════════════════════════════════════════════
# ALS: multi-term rank-1 ternary decomposition with per-block scales
# ═══════════════════════════════════════════════════════════════════════════

def als_decompose(W, num_terms=12, max_iter=10, qk=128):
    """
    Decompose W into sum of rank-1 ternary terms via ALS.

    Each term is an outer product u ⊗ v with u, v in {-1, 0, +1}, scaled by a
    PER-BLOCK float32 scale instead of a single global alpha: for each qk-wide
    column block b, s_b = sum(residual_b * T_b) / sum(T_b^2) (least squares,
    only over nonzero entries), then residual -= s_b * T_b per block.

    Returns:
        terms: list of (scales, T) where scales = list of per-block float32,
               T = ternary matrix (out_f, in_f) int8 (row-major, row=out feature)
        errs:  per-term cumulative relative errors (err after each term)
    """
    out_f, in_f = W.shape
    residual = W.float().clone()
    w_norm = torch.norm(W).item()
    n_blocks = (in_f + qk - 1) // qk
    terms = []
    errs = []

    for t in range(num_terms):
        # Power iteration for dominant singular triplet
        v = torch.randn(in_f)
        u = torch.randn(out_f)
        for _ in range(5):
            v = residual.T @ u
            v = v / (v.norm() + 1e-10)
            u = residual @ v
            u = u / (u.norm() + 1e-10)
        s = (u @ (residual @ v)).abs().item()

        # Round to ternary via alternating optimization
        u_tern = torch.sign(u).to(torch.int8)
        v_tern = torch.sign(v).to(torch.int8)

        for _ in range(max_iter):
            # Fix v, optimize u: ternary rounding of residual @ v
            proj_u = residual @ v_tern.float()
            u_new = torch.where(proj_u.abs() > 1e-6, torch.sign(proj_u).to(torch.int8), torch.tensor(0, dtype=torch.int8))
            # Fix u, optimize v
            proj_v = residual.T @ u_new.float()
            v_new = torch.where(proj_v.abs() > 1e-6, torch.sign(proj_v).to(torch.int8), torch.tensor(0, dtype=torch.int8))
            if torch.equal(u_new, u_tern) and torch.equal(v_new, v_tern):
                break
            u_tern, v_tern = u_new, v_new

        # Build ternary outer product
        outer = torch.outer(u_tern.float(), v_tern.float())  # (out_f, in_f)
        T_tern = torch.sign(outer).to(torch.int8)

        # Per-block least-squares scales (only over nonzero entries)
        scales = []
        for b in range(n_blocks):
            start = b * qk
            end = min(start + qk, in_f)
            Tb = T_tern[:, start:end]
            Rb = residual[:, start:end]
            mask = Tb != 0
            if mask.sum() > 0:
                s_b = (Rb[mask] * Tb[mask]).sum() / (Tb[mask] ** 2).sum()
            else:
                s_b = 0.0
            scales.append(float(s_b))

        # Update residual per block
        for b in range(n_blocks):
            start = b * qk
            end = min(start + qk, in_f)
            if scales[b] != 0:
                residual[:, start:end] -= scales[b] * T_tern[:, start:end].float()

        terms.append((scales, T_tern))
        errs.append(torch.norm(residual).item() / w_norm * 100 if w_norm > 0 else 0.0)

    return terms, errs

def pack_als_block_terms(terms_with_scales, qk=128):
    """Pack a list of (scales, T) ALS terms into the layer_type=2 container.

    Layout: [num_terms:u32][term0_len:u32][term0_data]...[termN_len:u32][termN_data]

    Each term_data is the I2_S per-row block layout:
      per row: [block0: 32 code bytes + 4 scale bytes] x n_blocks,
      n_blocks = ceil(in_f/128). Codes are 2-bit/weight, 4 vals/byte MSB-first,
      mapping +1->2, -1->3, 0->0.
    """
    buf = bytearray(struct.pack('<I', len(terms_with_scales)))
    for scales, tv in terms_with_scales:
        blob = pack_als_block(tv, scales, qk)
        buf.extend(struct.pack('<I', len(blob)))
        buf.extend(blob)
    return bytes(buf)

def pack_als_block(tv_tensor, scales, qk=128):
    """Pack one ALS term (ternary matrix + per-block scales) into the per-row block blob.

    Per row: for each qk-wide block: qk/4 code bytes (2-bit, MSB-first) + 4 scale bytes.
    Codes map +1->2, -1->3, 0->0; partial tail blocks are zero-padded (0 = zero weight).
    """
    out_f, in_f = tv_tensor.shape
    n_blocks = len(scales)
    codes_per_block = qk // 4

    # Vectorized ternary -> 2-bit code mapping (+1->2, -1->3, 0->0)
    tv = tv_tensor.to(torch.int8)
    codes = torch.zeros_like(tv, dtype=torch.uint8)
    codes.masked_fill_(tv == 1, 2)
    codes.masked_fill_(tv == -1, 3)

    buf = bytearray()
    for row in range(out_f):
        for b in range(n_blocks):
            start = b * qk
            end = min(start + qk, in_f)
            block = codes[row, start:end]
            if end - start < qk:
                block = torch.nn.functional.pad(block, (0, qk - (end - start)))
            # Pack 4 vals per byte MSB-first: [a,b,c,d] -> (a<<6)|(b<<4)|(c<<2)|d
            c4 = block.reshape(-1, 4)
            packed = (c4[:, 0] << 6) | (c4[:, 1] << 4) | (c4[:, 2] << 2) | c4[:, 3]
            buf.extend(packed.cpu().numpy().tobytes())
            buf.extend(struct.pack('<f', float(scales[b])))
    return bytes(buf)

def export_als_blocks(out_dir, model_name, num_terms=12):
    """ALS export: multi-term rank-1 ternary decomposition with per-block scales.

    Writes model_decomposed.bin (magic 0xDEADBEEF). Layer types:
      0 = TQ1 legacy (unused, backward compat), 1 = RAW_FP32, 2 = multi-term block container.
    """
    from transformers import AutoModelForCausalLM
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_bin = out_dir / 'model_decomposed.bin'

    print("=" * 70)
    print(f"EXPORT: {model_name} -> ALS per-block format ({num_terms} terms)")
    print(f"Output: {out_dir}")
    print("=" * 70)

    print(f"\nDownloading {model_name} from HuggingFace...")
    model_hf = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=DTYPE).eval()

    q_layers = []
    total_fp32 = 0
    total_als = 0
    n_quantized = 0
    n_skipped = 0

    print(f"\n[ALS decomposition ({num_terms} terms per layer, per-block scales)...]\n")
    t0 = time.time()

    all_layers = get_model_layers(model_hf, model_hf.config)
    total_layers = len(all_layers)
    done_count = 0

    for name, W in all_layers:
        done_count += 1
        print(f"[PROGRESS] {100.0 * done_count / total_layers:.0f}%")
        out_f, in_f = W.shape
        fp32_bytes = out_f * in_f * 4

        if not should_quantize(name):
            raw_data = W.flatten().numpy().tobytes()
            q_layers.append({
                'name': name, 'out_features': out_f, 'in_features': in_f,
                'layer_type': 1, 'data': raw_data, 'err': -1.0,
            })
            print(f"  - {name:50s} RAW FP32 ({len(raw_data)/1e6:.1f} MB)")
            n_skipped += 1
            continue

        t1 = time.time()
        terms, errs = als_decompose(W, num_terms=num_terms)

        # Pack into layer_type=2 multi-term container blob
        blob = pack_als_block_terms(terms, 128)
        err = errs[-1]
        t2 = time.time()
        total_fp32 += fp32_bytes
        total_als += len(blob)

        q_layers.append({
            'name': name, 'out_features': out_f, 'in_features': in_f,
            'layer_type': 2, 'data': blob, 'err': err,
        })
        ratio = fp32_bytes / len(blob)
        status = 'OK' if err < 20 else '??'
        err_chain = ' -> '.join(f'{e:.1f}' for e in errs)
        print(f"  {status} {name:48s} [{out_f:5d},{in_f:5d}] "
              f"err={err_chain}%  comp={ratio:5.1f}x  {t2-t1:.2f}s")
        n_quantized += 1

    t_quant = time.time() - t0

    # Write ALS binary: magic 0xDEADBEEF, then per-layer name/type/data
    print(f"\n[Writing {model_bin}...]")
    with open(model_bin, 'wb') as f:
        f.write(struct.pack('<I', 0xDEADBEEF))  # ALS magic
        f.write(struct.pack('<I', len(q_layers)))
        for layer in q_layers:
            name_bytes = layer['name'].encode('utf-8')
            f.write(struct.pack('<I', len(name_bytes)))
            f.write(name_bytes)
            f.write(struct.pack('<II', layer['out_features'], layer['in_features']))
            f.write(struct.pack('<B', layer['layer_type']))
            f.write(struct.pack('<I', len(layer['data'])))
            f.write(layer['data'])

    file_size = os.path.getsize(model_bin)
    total_raw = sum(len(l['data']) for l in q_layers if l['layer_type'] == 1)

    print("\n" + "=" * 70)
    print(f"COMPRESSION METRICS (ALS per-block {num_terms} terms)")
    print("=" * 70)
    print(f"  ALS quantized layers:       {n_quantized}")
    print(f"  Raw FP32 layers:            {n_skipped}")
    print(f"  Ternary data:               {total_als / 1e6:.1f} MB")
    print(f"  Binary file:                {file_size / 1e6:.1f} MB")
    total_comp = total_als + total_raw
    if total_fp32 > 0:
        print(f"  Compression ratio vs FP32:  {total_fp32 / total_comp:.1f}x")
    print(f"  ALS time:                   {t_quant:.1f}s")
    print(f"\nDone! Model saved to {model_bin}")
    return 0


# ═══════════════════════════════════════════════════════════════════════════
# model_extra.bin writer (config + embedding + RMS norms)
# ═══════════════════════════════════════════════════════════════════════════

def write_extra(out_dir, model_hf):
    """
    Write model_extra.bin containing:
      - 9 × int32/float32 config fields
      - embedding table [vocab_size, hidden_size] float32
      - final_norm [hidden_size] float32
      - per-layer: [input_layernorm, post_attention_layernorm] each [hidden_size] float32
    """
    cfg = model_hf.config
    extra_path = Path(out_dir) / 'model_extra.bin'

    # 9 config fields matching loader.h read order
    fields = [
        ('vocab_size', cfg.vocab_size, 'i'),
        ('hidden_size', cfg.hidden_size, 'i'),
        ('intermediate_size', cfg.intermediate_size, 'i'),
        ('num_hidden_layers', cfg.num_hidden_layers, 'i'),
        ('num_attention_heads', cfg.num_attention_heads, 'i'),
        ('num_key_value_heads', cfg.num_key_value_heads, 'i'),
        ('rms_norm_eps', cfg.rms_norm_eps, 'f'),
        ('rope_theta', getattr(cfg, 'rope_theta', 10000.0), 'f'),
        ('max_position_embeddings', cfg.max_position_embeddings, 'i'),
    ]

    with open(extra_path, 'wb') as f:
        for name, val, fmt in fields:
            if fmt == 'i':
                f.write(struct.pack('<i', int(val)))
            else:
                f.write(struct.pack('<f', float(val)))

        # Embedding weights
        emb = model_hf.get_input_embeddings().weight.data.float()
        f.write(emb.numpy().tobytes())

        # Final norm
        if hasattr(model_hf.model, 'norm'):
            fn = model_hf.model.norm.weight.data.float()
        elif hasattr(model_hf.model, 'final_norm'):
            fn = model_hf.model.final_norm.weight.data.float()
        else:
            print('  Warning: could not find final norm, writing zeros')
            fn = torch.zeros(cfg.hidden_size)
        f.write(fn.numpy().tobytes())

        # Per-layer norms
        for i in range(cfg.num_hidden_layers):
            layer = model_hf.model.layers[i]
            in_ln = layer.input_layernorm.weight.data.float()
            pa_ln = layer.post_attention_layernorm.weight.data.float()
            f.write(in_ln.numpy().tobytes())
            f.write(pa_ln.numpy().tobytes())

    return extra_path


def main():
    parser = argparse.ArgumentParser(description='Export HF model to Terllama ALS binary format')
    parser.add_argument('--model', default='HuggingFaceTB/SmolLM2-135M',
                        help='HuggingFace model name')
    parser.add_argument('--outdir', default='',
                        help='Output directory (default: ~/.terllama/models/<slug>)')
    parser.add_argument('--terms', type=int, default=12,
                        help='Number of ALS terms (default: 12)')
    parser.add_argument('--rotate', type=int, default=0,
                        help='RoPE theta (-1: max positional, 0: no change, N: specific value)')
    args = parser.parse_args()

    model_slug = args.model.replace('/', '-')
    out_dir = args.outdir or str(Path.home() / '.terllama' / 'models' / model_slug)

    ret = export_als_blocks(out_dir, args.model, num_terms=args.terms)
    if ret != 0:
        return ret

    # Write model_extra.bin — reload model if needed (it's cached, fast)
    print('\n[Writing model_extra.bin...]')
    from transformers import AutoModelForCausalLM as _M
    m = _M.from_pretrained(args.model, dtype=torch.float32).eval()
    extra_path = write_extra(out_dir, m)
    print(f'  Wrote {extra_path} ({os.path.getsize(extra_path)/1e6:.1f} MB)')

    # Write tokenizer files (tokenizer.json + tokenizer_config.json + extras)
    # so the C++ loader can find model_dir/tokenizer.json at runtime.
    print('\n[Writing tokenizer files...]')
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.save_pretrained(out_dir)
    tok_json = os.path.join(out_dir, 'tokenizer.json')
    print(f'  Wrote tokenizer -> {out_dir} '
          f'({"tokenizer.json: %.1f MB" % (os.path.getsize(tok_json)/1e6) if os.path.exists(tok_json) else "no tokenizer.json"})')
    return 0

if __name__ == '__main__':
    sys.exit(main())
