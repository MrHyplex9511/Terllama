"""
Export HF model -> ALS per-block binary for Terllama C++ inference.

ALS (only format): greedy element-wise ternary decomposition with iterative
refinement (each term = full ternary matrix with a power-of-2 scale, refined
by re-solving one term against the residual of all others), then per-block
float32 scales (128-wide blocks, 2 bits/elem) fitted by least squares.

This matches utils/phase4_ppl.py (the algorithm that produced the README
benchmarks: SmolLM2 PPL 15.89 -> 16.84 @ 8/10/12/15 terms, FFN 4.92% err).
NOTE: the old rank-1 outer-product ALS was mathematically rank-limited
(SVD rank-12 floor ~60% err on dense weights) and could never reach those
numbers; it is replaced by this element-wise variant.

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
# ALS: greedy element-wise ternary decomposition + iterative refinement
# (ported from utils/phase4_ppl.py — the algorithm behind the README numbers)
# ═══════════════════════════════════════════════════════════════════════════

def _greedy_terms(W, num_terms):
    """Greedy element-wise ternary decomposition.

    Each term: scale a = 2^k (power of two), ternary matrix T = clamp(round(R/a), -1, 1).
    No rank constraint — every element of the residual is rounded independently,
    so the decomposition is NOT limited by the singular-value spectrum.
    Returns list of (a, T) with a float scale, T int8 ternary (out_f, in_f).
    """
    R, terms = W.float().clone(), []
    for _ in range(num_terms):
        r_max = R.abs().max().item()
        if r_max < 1e-8:
            terms.append((0.0, torch.zeros_like(R, dtype=torch.int8)))
            continue
        # Search power-of-2 scale a = 2^k around the residual magnitude
        lo = int(math.floor(math.log2(max(r_max / 3, 1e-10))))
        hi = int(math.ceil(math.log2(max(r_max * 1.5, 1e-10)))) + 2
        best_k, best_e = None, float('inf')
        for k in range(lo, hi):
            a = 2.0 ** k
            Tc = torch.clamp(torch.round(R / a), -1.0, 1.0)
            e = torch.norm(R - a * Tc).item()
            if e < best_e:
                best_e, best_k = e, k
        a = 2.0 ** best_k
        T = torch.clamp(torch.round(R / a), -1.0, 1.0).to(torch.int8)
        R -= a * T.float()
        terms.append((float(a), T))
    return terms

def als_decompose(W, num_terms=12, max_iter=5, qk=128):
    """
    Decompose W into num_terms element-wise ternary terms, then refine.

    Phase 1 (init): greedy power-of-2 ternary rounding (see _greedy_terms).
    Phase 2 (refine): for each term, rebuild the residual R = W - sum_{j!=i} a_j*T_j
        and re-solve term i against R (best power-of-2 scale + ternary rounding).
        Repeating this coordinate descent drives each term toward its optimal
        element-wise ternary fit.

    Returns:
        terms: list of (scales, T) where scales = list of per-block float32
               (n_blocks = ceil(in_f/qk), fitted by least squares per block),
               T = ternary matrix (out_f, in_f) int8 (row-major, row=out feature)
        errs:  per-term cumulative relative errors (err after each term)
    """
    out_f, in_f = W.shape
    W = W.float()
    w_norm = torch.norm(W).item()
    n_blocks = (in_f + qk - 1) // qk

    terms = _greedy_terms(W, num_terms)
    for _ in range(max_iter):
        for i in range(num_terms):
            # Residual excluding term i
            R = W.clone()
            for j in range(num_terms):
                if j != i and terms[j][0] != 0.0:
                    R -= terms[j][0] * terms[j][1].float()
            r_max = R.abs().max().item()
            if r_max < 1e-8:
                terms[i] = (0.0, torch.zeros_like(terms[i][1]))
                continue
            lo = int(math.floor(math.log2(max(r_max / 3, 1e-10))))
            hi = int(math.ceil(math.log2(max(r_max * 1.5, 1e-10)))) + 2
            best_k, best_e = None, float('inf')
            for k in range(lo, hi):
                a = 2.0 ** k
                Tc = torch.clamp(torch.round(R / a), -1.0, 1.0)
                e = torch.norm(R - a * Tc).item()
                if e < best_e:
                    best_e, best_k = e, k
            a = 2.0 ** best_k
            terms[i] = (float(a), torch.clamp(torch.round(R / a), -1.0, 1.0).to(torch.int8))

    # Joint per-block least-squares scales: for each block, solve
    #   min || W_block - sum_i s_bi * T_bi ||^2
    # over ALL term scales s_bi simultaneously (normal equations, NxN per block).
    # This keeps the refined joint optimum (global-alpha error ~5%) and can only
    # improve it; a sequential forward pass would NOT — terms are only meaningful
    # together, not one-at-a-time.
    N = num_terms
    flat_T = [t[1].float() for t in terms]  # (out_f, in_f) each
    out_terms = []
    errs = []
    residual = W.clone()
    for b in range(n_blocks):
        start = b * qk
        end = min(start + qk, in_f)
        Wb = W[:, start:end]                        # (out_f, bw)
        # Build A = (out_f*bw, N), column i = flattened ternary block i
        cols = [Tb[:, start:end].reshape(-1) for Tb in flat_T]
        A = torch.stack(cols, dim=1)                # (out_f*bw, N)
        if A.numel() == 0:
            scales_b = [0.0] * N
        else:
            # Solve NxN normal equations in float64 with an adaptive ridge.
            # (For sparse ternary patterns AtA can be ill-conditioned up to
            # 1e34; SVD-based lstsq on the full tall matrix silently returns
            # a zero solution for very large layers, e.g. lm_head 49152 rows,
            # so we go through the tiny normal system instead.)
            A64 = A.double()
            AtA = A64.t() @ A64
            Atw = A64.t() @ Wb.reshape(-1).double()
            ridge = 1e-10 * torch.diag(AtA).max().item()
            try:
                s_b = torch.linalg.solve(AtA + ridge * torch.eye(N, dtype=torch.float64), Atw)
            except Exception:
                s_b = torch.linalg.lstsq(A64, Wb.reshape(-1).double(), driver='gelsd').solution
            scales_b = [float(x) for x in s_b]
        out_terms.append(scales_b)

    # Transpose out_terms: (per-block lists) -> (per-term lists of n_blocks scales)
    per_term_scales = []
    for i in range(N):
        per_term_scales.append([out_terms[b][i] for b in range(n_blocks)])

    # Rebuild residual with the joint-scaled terms for the error chain
    residual = W.clone()
    errs = []
    for i in range(N):
        Ti = flat_T[i]
        for b in range(n_blocks):
            start = b * qk
            end = min(start + qk, in_f)
            sb = per_term_scales[i][b]
            if sb != 0.0:
                residual[:, start:end] -= sb * Ti[:, start:end]
        errs.append(torch.norm(residual).item() / w_norm * 100 if w_norm > 0 else 0.0)

    out_terms = list(zip(per_term_scales, [t[1] for t in terms]))
    return out_terms, errs

def pack_als_block_terms(terms_with_scales, qk=128):
    """Pack a list of (scales, T) ALS terms into the layer_type=2 container.

    Layout: [num_terms:u32][term0_len:u32][term0_data]...[termN_len:u32][termN_data]

    Each term_data is the I2_S per-row block layout:
      per row: [block0: 32 code bytes + 4 scale bytes] x n_blocks,
      n_blocks = ceil(in_f/128). Codes are 2-bit/weight, 4 vals/byte MSB-first,
      mapping {-1,0,+1} -> {0,1,2} (legacy I2S, matches all C++ decoders).
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
    Encoding matches the C++ decoders (all four kernels use legacy I2S mapping):
        {-1, 0, +1} -> {0, 1, 2}; code 3 is never emitted.
    Tail blocks are zero-padded with code 1 (ternary 0).
    """
    out_f, in_f = tv_tensor.shape
    n_blocks = len(scales)
    codes_per_block = qk // 4

    # Vectorized ternary -> 2-bit code mapping (-1->0, 0->1, +1->2)
    tv = tv_tensor.to(torch.int8)
    codes = torch.ones_like(tv, dtype=torch.uint8)  # default 0 -> code 1
    codes.masked_fill_(tv == -1, 0)
    codes.masked_fill_(tv == 1, 2)

    buf = bytearray()
    for row in range(out_f):
        for b in range(n_blocks):
            start = b * qk
            end = min(start + qk, in_f)
            block = codes[row, start:end]
            if end - start < qk:
                # Pad with code 1 (ternary 0) — never 0 which means -1.
                block = torch.nn.functional.pad(block, (0, qk - (end - start)), value=1)
            # Pack 4 vals per byte MSB-first: [a,b,c,d] -> (a<<6)|(b<<4)|(c<<2)|d
            c4 = block.reshape(-1, 4)
            packed = (c4[:, 0] << 6) | (c4[:, 1] << 4) | (c4[:, 2] << 2) | c4[:, 3]
            buf.extend(packed.cpu().numpy().tobytes())
            buf.extend(struct.pack('<f', float(scales[b])))
    return bytes(buf)

def export_als_blocks(out_dir, model_name, num_terms=12):
    """ALS export: greedy element-wise ternary decomposition + refinement, per-block scales.

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

    # Count Qwen3 Q/K RMSNorm pseudo-layers for accurate progress reporting.
    n_qk_norm_total = 0
    if hasattr(model_hf, 'model') and hasattr(model_hf.model, 'layers'):
        first_attn = model_hf.model.layers[0].self_attn
        for norm_attr in ('q_norm', 'k_norm'):
            if hasattr(first_attn, norm_attr):
                n_qk_norm_total += len(model_hf.model.layers)

    total_layers = len(all_layers) + n_qk_norm_total
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

    # ── Qwen3-style per-head Q/K RMSNorm (Qwen3RMSNorm, not nn.Linear) ──
    # The main loop only walks nn.Linear modules, so q_norm/k_norm (weight
    # shape [head_dim], shared across heads) are invisible there. Export them
    # as RAW_FP32 pseudo-layers so the loader/inference can apply them.
    # Names mirror the transformer_block lookup: model.layers.<N>.self_attn.q_norm.
    n_qk_norm = 0
    if hasattr(model_hf, 'model') and hasattr(model_hf.model, 'layers'):
        first_attn = model_hf.model.layers[0].self_attn
        for norm_attr in ('q_norm', 'k_norm'):
            if not hasattr(first_attn, norm_attr):
                continue
            for li in range(len(model_hf.model.layers)):
                w = getattr(model_hf.model.layers[li].self_attn, norm_attr).weight.data
                raw_data = w.flatten().to(dtype=DTYPE).numpy().tobytes()
                done_count += 1
                print(f"[PROGRESS] {100.0 * done_count / total_layers:.0f}%")
                q_layers.append({
                    'name': f'model.layers.{li}.self_attn.{norm_attr}',
                    'out_features': w.shape[0], 'in_features': w.shape[0],
                    'layer_type': 1, 'data': raw_data, 'err': -1.0,
                })
                n_qk_norm += 1
        if n_qk_norm:
            print(f"  - Exported {n_qk_norm} Qwen3 Q/K RMSNorm weights as RAW FP32")

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
