"""
Export HF model -> I2_S or ALS binary for Terllama C++ inference.

I2_S (--format i2s): BitNet mean-scale, 4 vals/byte + per-block scales.
ALS  (--format als): Alternating Least Squares, multi-term ternary, 2 bits/elem.

Usage:
  python scripts/export_ternary_model_bitnet.py --model HuggingFaceTB/SmolLM2-135M --format als
"""
import argparse, torch, math, time, struct, json, os, sys
from pathlib import Path

torch.manual_seed(42)
DTYPE = torch.float32
QUANTIZED_LAYERS = {'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'}

def should_quantize(layer_name):
    return any(x in layer_name for x in QUANTIZED_LAYERS)

# ═══════════════════════════════════════════════════════════════════════════
# I2_S: mean-block quantization, 4 vals/byte
# ═══════════════════════════════════════════════════════════════════════════

def quantize_weights_mean_block(weight, block_size=128):
    w = weight.float()
    out_f, in_f = w.shape
    pad = (block_size - in_f % block_size) % block_size
    if pad > 0:
        w = torch.nn.functional.pad(w, (0, pad))
    n_blocks = w.shape[1] // block_size
    codes = torch.zeros(out_f, w.shape[1], dtype=torch.int8)
    scales = torch.zeros(out_f, n_blocks, dtype=torch.float32)
    for i in range(out_f):
        for b in range(n_blocks):
            block = w[i, b*block_size:(b+1)*block_size]
            s = 1.0 / block.abs().mean().clamp(min=1e-5)
            codes[i, b*block_size:(b+1)*block_size] = \
                (block * s).round().clamp(-1, 1).to(torch.int8)
            scales[i, b] = s
    if pad > 0:
        codes = codes[:, :in_f]
    return codes, scales

def pack_i2s_with_scales(codes, scales, qk=128):
    """Pack ternary codes into I2_S: 4 vals/byte + per-block float32 scale."""
    out_f, in_f = codes.shape
    n_blocks = scales.shape[1]
    codes_per_block = qk // 4
    buf = bytearray()
    for row in range(out_f):
        row_codes = codes[row].tolist()
        for b in range(n_blocks):
            block_start = b * qk
            block_end = min(block_start + qk, in_f)
            byte_buf = bytearray(codes_per_block)
            for j in range(0, qk, 4):
                byte_val = 0
                for k in range(4):
                    idx = block_start + j + k
                    if idx < in_f:
                        val = row_codes[idx] + 1  # -1→0, 0→1, 1→2
                    else:
                        val = 0
                    byte_val |= (val & 0x03) << (6 - k * 2)
                byte_buf[j // 4] = byte_val
            buf.extend(byte_buf)
            buf.extend(struct.pack('<f', scales[row, b].item()))
    return bytes(buf)

# ═══════════════════════════════════════════════════════════════════════════
# ALS: multi-term rank-1 ternary decomposition
# ═══════════════════════════════════════════════════════════════════════════

def als_decompose(W, num_terms=8, max_iter=10):
    """
    Decompose W into sum of rank-1 ternary terms via ALS.

    Each term: alpha * outer(u, v) where u, v in {-1, 0, +1}.
    Returns list of (alpha, T) where T is ternary matrix flattened row-major.
    """
    out_f, in_f = W.shape
    residual = W.float().clone()
    terms = []

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

        # Build outer product
        outer = torch.outer(u_tern.float(), v_tern.float())  # (out_f, in_f)

        # Optimal alpha: least-squares scale
        mask = outer != 0
        if mask.sum() > 0:
            alpha = (residual[mask] * outer[mask]).sum() / (outer[mask] ** 2).sum()
        else:
            alpha = 0.0

        # Build full ternary matrix (stored packed per term)
        T = torch.zeros(out_f * in_f, dtype=torch.int8)
        if alpha != 0:
            outer_tern = torch.sign(outer).to(torch.int8)
            T = outer_tern.flatten()

        terms.append((alpha, T))
        residual -= alpha * outer

    return terms

def pack_als_ternary(alpha, tv_tensor):
    """
    Vectorized ALS packing. PyTorch ops, no Python loops.

    Maps: +1→bits 10(2), -1→bits 11(3), 0→bits 00(0).
    Packs 4 values/byte MSB-first.
    """
    # Map ternary -> 2-bit codes
    tv = tv_tensor.to(torch.int8)
    codes = torch.zeros(tv.numel(), dtype=torch.uint8, device=tv.device)
    codes.masked_fill_(tv == 1, 2)
    codes.masked_fill_(tv == -1, 3)

    # Pad to multiple of 4
    n = codes.shape[0]
    pad = (4 - n % 4) % 4
    if pad:
        codes = torch.nn.functional.pad(codes, (0, pad))

    # Pack 4 vals per byte: [a,b,c,d] -> (a<<6)|(b<<4)|(c<<2)|d
    c4 = codes.reshape(-1, 4)
    packed = (c4[:, 0] << 6) | (c4[:, 1] << 4) | (c4[:, 2] << 2) | c4[:, 3]

    return struct.pack('<f', alpha) + packed.cpu().numpy().tobytes()

def export_als(out_dir, model_name, num_terms=8):
    """ALS export: multi-term rank-1 ternary decomposition."""
    from transformers import AutoModelForCausalLM
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_bin = out_dir / 'model_decomposed.bin'

    print("=" * 70)
    print(f"EXPORT: {model_name} -> ALS format ({num_terms} terms)")
    print(f"Output: {out_dir}")
    print("=" * 70)

    print(f"\nDownloading {model_name} from HuggingFace...")
    model_hf = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=DTYPE).eval()

    q_layers = []
    total_fp32 = 0
    total_als = 0
    n_quantized = 0
    n_skipped = 0

    print(f"\n[ALS decomposition ({num_terms} terms per layer)...]\n")
    t0 = time.time()

    for name, mod in model_hf.named_modules():
        if not isinstance(mod, torch.nn.Linear):
            continue
        W = mod.weight.data.to(dtype=DTYPE)
        out_f, in_f = W.shape
        fp32_bytes = out_f * in_f * 4

        if not should_quantize(name):
            raw_data = W.flatten().numpy().tobytes()
            q_layers.append({
                'name': name, 'out_features': out_f, 'in_features': in_f,
                'data': raw_data, 'err': -1.0, 'is_raw': True,
            })
            print(f"  - {name:50s} RAW FP32 ({len(raw_data)/1e6:.1f} MB)")
            n_skipped += 1
            continue

        t1 = time.time()
        terms = als_decompose(W, num_terms=num_terms)

        # Pack each term: alpha + 2-bit ternary data (vectorized)
        packed_terms = []
        for alpha, tv in terms:
            packed_terms.append(pack_als_ternary(alpha, tv))

        # Reconstruct for error (vectorized)
        W_hat = torch.zeros(out_f, in_f)
        for alpha, tv in terms:
            if alpha != 0:
                W_hat += alpha * tv.reshape(out_f, in_f).float()

        err = torch.norm(W - W_hat).item() / torch.norm(W).item() * 100
        t2 = time.time()
        als_size = sum(len(p) for p in packed_terms)
        total_fp32 += fp32_bytes
        total_als += als_size

        q_layers.append({
            'name': name, 'out_features': out_f, 'in_features': in_f,
            'data': packed_terms, 'err': err, 'is_raw': False,
        })
        ratio = fp32_bytes / als_size
        status = 'OK' if err < 20 else '??'
        print(f"  {status} {name:48s} [{out_f:5d},{in_f:5d}] "
              f"err={err:5.2f}%  comp={ratio:5.1f}x  {t2-t1:.2f}s")
        n_quantized += 1

    t_quant = time.time() - t0

    # Write ALS binary
    print(f"\n[Writing {model_bin}...]")
    with open(model_bin, 'wb') as f:
        f.write(struct.pack('<I', 0xDEADBEEF))  # ALS magic
        f.write(struct.pack('<I', len(q_layers)))
        for layer in q_layers:
            name_bytes = layer['name'].encode('utf-8')
            f.write(struct.pack('<I', len(name_bytes)))
            f.write(name_bytes)
            f.write(struct.pack('<II', layer['out_features'], layer['in_features']))
            if layer.get('is_raw', False):
                f.write(struct.pack('<I', 0))  # num_terms=0 signals raw FP32
                f.write(struct.pack('<I', len(layer['data'])))  # data_len
                f.write(layer['data'])
            else:
                f.write(struct.pack('<I', len(layer['data'])))  # num_terms
                for term_data in layer['data']:
                    f.write(term_data)

    file_size = os.path.getsize(model_bin)
    total_raw = sum(len(l['data']) for l in q_layers if l.get('is_raw', False))

    print("\n" + "=" * 70)
    print(f"COMPRESSION METRICS (ALS {num_terms} terms)")
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


def export_i2s(out_dir, model_name):
    """I2_S export: mean-block ternary quantization."""
    from transformers import AutoModelForCausalLM
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_bin = out_dir / 'model_decomposed_i2s.bin'

    print("=" * 70)
    print(f"EXPORT: {model_name} -> I2_S BitNet format")
    print(f"Output: {out_dir}")
    print("=" * 70)

    print(f"\nDownloading {model_name} from HuggingFace...")
    model_hf = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=DTYPE).eval()

    q_layers = []
    total_fp32 = 0
    total_i2s = 0
    n_quantized = 0
    n_skipped = 0

    print("\n[Quantizing layers (BitNet style)...]\n")
    t0 = time.time()

    for name, mod in model_hf.named_modules():
        if not isinstance(mod, torch.nn.Linear):
            continue
        if not should_quantize(name):
            W = mod.weight.data.to(dtype=DTYPE)
            out_f, in_f = W.shape
            raw_data = W.flatten().numpy().tobytes()
            q_layers.append({
                'name': name, 'out_features': out_f, 'in_features': in_f,
                'data': raw_data, 'err': -1.0, 'is_raw': True,
            })
            print(f"  - {name:50s} RAW FP32 ({len(raw_data)/1e6:.1f} MB)")
            n_skipped += 1
            continue

        W = mod.weight.data.to(dtype=DTYPE)
        out_f, in_f = W.shape
        fp32_bytes = out_f * in_f * 4

        t1 = time.time()
        codes, scales = quantize_weights_mean_block(W, 128)
        t2 = time.time()

        i2s_data = pack_i2s_with_scales(codes, scales, 128)

        W_hat = codes.float() / (1.0 / W.float().abs().mean().clamp(min=1e-5))
        err = torch.norm(W - W_hat).item() / torch.norm(W).item() * 100

        total_fp32 += fp32_bytes
        total_i2s += len(i2s_data)

        q_layers.append({
            'name': name, 'out_features': out_f, 'in_features': in_f,
            'data': i2s_data, 'err': err, 'is_raw': False,
        })
        ratio = fp32_bytes / len(i2s_data)
        status = 'OK' if err < 20 else '??'
        print(f"  {status} {name:48s} [{out_f:5d},{in_f:5d}] "
              f"err={err:5.2f}%  comp={ratio:5.1f}x  {t2-t1:.2f}s")
        n_quantized += 1

    t_quant = time.time() - t0

    print(f"\n[Writing {model_bin}...]")
    with open(model_bin, 'wb') as f:
        f.write(struct.pack('<I', 0x5F533249))  # magic: I2S_
        f.write(struct.pack('<I', len(q_layers)))
        for layer in q_layers:
            name_bytes = layer['name'].encode('utf-8')
            f.write(struct.pack('<I', len(name_bytes)))
            f.write(name_bytes)
            f.write(struct.pack('<II', layer['out_features'], layer['in_features']))
            layer_type = 1 if layer.get('is_raw', False) else 0
            f.write(struct.pack('<B', layer_type))
            f.write(struct.pack('<I', len(layer['data'])))
            f.write(layer['data'])

    file_size = os.path.getsize(model_bin)
    total_raw = sum(len(l['data']) for l in q_layers if l.get('is_raw', False))

    print("\n" + "=" * 70)
    print("COMPRESSION METRICS (BitNet I2_S)")
    print("=" * 70)
    print(f"  I2_S quantized layers:       {n_quantized}")
    print(f"  Raw FP32 layers:             {n_skipped}")
    print(f"  Original FP32 (quant only):  {total_fp32 / 1e6:.1f} MB")
    print(f"  I2_S compressed:             {total_i2s / 1e6:.1f} MB")
    print(f"  Raw FP32 data:               {total_raw / 1e6:.1f} MB")
    print(f"  Binary file:                 {file_size / 1e6:.1f} MB")
    total_comp = total_i2s + total_raw
    if total_fp32 > 0:
        print(f"  Compression ratio vs FP32:   {total_fp32 / total_comp:.1f}x")
        print(f"  Bits per param (I2_S only):  {total_i2s * 8 / (total_fp32 / 4):.2f}")
    print(f"  Quantization time:           {t_quant:.1f}s")
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
    parser = argparse.ArgumentParser(description='Export HF model to Terllama binary format')
    parser.add_argument('--model', default='HuggingFaceTB/SmolLM2-135M',
                        help='HuggingFace model name')
    parser.add_argument('--outdir', default='',
                        help='Output directory (default: ~/.terllama/models/<slug>)')
    parser.add_argument('--format', choices=['i2s', 'als'], default='i2s',
                        help='Export format (default: i2s)')
    parser.add_argument('--terms', type=int, default=8,
                        help='Number of ALS terms (default: 8, only for --format als)')
    parser.add_argument('--rotate', type=int, default=0,
                        help='RoPE theta (-1: max positional, 0: no change, N: specific value)')
    args = parser.parse_args()

    model_slug = args.model.replace('/', '-')
    out_dir = args.outdir or str(Path.home() / '.terllama' / 'models' / model_slug)

    if args.format == 'als':
        ret = export_als(out_dir, args.model, num_terms=args.terms)
    else:
        ret = export_i2s(out_dir, args.model)
    if ret != 0:
        return ret

    # Write model_extra.bin — reload model if needed (it's cached, fast)
    print('\n[Writing model_extra.bin...]')
    from transformers import AutoModelForCausalLM as _M
    m = _M.from_pretrained(args.model, dtype=torch.float32).eval()
    extra_path = write_extra(out_dir, m)
    print(f'  Wrote {extra_path} ({os.path.getsize(extra_path)/1e6:.1f} MB)')
    return 0

if __name__ == '__main__':
    sys.exit(main())
