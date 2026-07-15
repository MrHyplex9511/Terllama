"""
Export SmolLM2-135M → I2_S binary. Terllama C++ inference.
BitNet: mean scaling, selective layer quant, I2_S packing.
"""
import torch, math, time, struct, json, os
from pathlib import Path

torch.manual_seed(42)
DTYPE = torch.float32

# BitNet: quantize 7 projection layers per block
QUANTIZED_LAYERS = {'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'}

def should_quantize(layer_name):
    return any(x in layer_name for x in QUANTIZED_LAYERS)

# Mean-Based Scaling (BitNet style)
def quantize_weights_mean_block(weight, block_size=128):
    """Block-wise mean ternary quant. Returns codes + scales."""
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

# I2_S: 4 ternary vals/byte, row-interleaved with scales
def pack_i2s_with_scales(codes, scales, qk=128):
    """Pack I2_S with per-block scales. Row by row.

    Per row: [blk0_codes(32B)][blk0_scale(4B)]...
    C++ decodes one row at a time. No index arithmetic.

    Args:
        codes: int8 tensor {-1,0,+1} (out_f, in_f)
        scales: float32 tensor (out_f, n_blocks)
        qk: block size (default 128)

    Returns:
        Bytes: interleaved block codes + scales
    """
    import struct
    out_f, in_f = codes.shape
    n_blocks = scales.shape[1]
    codes_per_block = qk // 4  # 32 bytes per block
    buf = bytearray()
    for row in range(out_f):
        row_codes = codes[row].tolist()
        for b in range(n_blocks):
            block_start = b * qk
            block_end = min(block_start + qk, in_f)
            # Pack qk vals → codes_per_block bytes
            byte_buf = bytearray(codes_per_block)
            for j in range(0, qk, 4):
                byte_val = 0
                for k in range(4):
                    idx = block_start + j + k
                    if idx < in_f:
                        val = row_codes[idx] + 1  # {-1,0,+1} -> {0,1,2}
                    else:
                        val = 0
                    byte_val |= (val & 0x03) << (6 - k * 2)
                byte_buf[j // 4] = byte_val
            buf.extend(byte_buf)
            buf.extend(struct.pack('<f', scales[row, b].item()))
    return bytes(buf)

def main():
    print("=" * 70)
    print("EXPORT: SmolLM2-135M to I2_S BitNet format")
    print("=" * 70)

    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        "HuggingFaceTB/SmolLM2-135M", torch_dtype=DTYPE
    ).eval()

    out_dir = Path("/media/extra/Symlinks/BitNet/utils/export")
    out_dir.mkdir(exist_ok=True)
    bin_path = out_dir / "model_decomposed_i2s.bin"

    q_layers = []
    total_fp32_bytes = 0
    total_i2s_bytes = 0

    print("\n[Quantizing layers (BitNet style)...]\n")
    t0 = time.time()
    n_quantized = 0
    n_skipped = 0

    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.Linear):
            continue
        if not should_quantize(name):
            # Raw FP32 for non-quantized layers (e.g. lm_head)
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

        # Reconstruction error
        W_hat = codes.float() / (1.0 / W.float().abs().mean().clamp(min=1e-5))
        err = torch.norm(W - W_hat).item() / torch.norm(W).item() * 100

        total_fp32_bytes += fp32_bytes
        total_i2s_bytes += len(i2s_data)

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

    print(f"\n[Writing {bin_path}...]")
    with open(bin_path, 'wb') as f:
        f.write(struct.pack('<I', 0x5F533249))  # magic: I2S_
        f.write(struct.pack('<I', len(q_layers)))
        for layer in q_layers:
            name_bytes = layer['name'].encode('utf-8')
            f.write(struct.pack('<I', len(name_bytes)))
            f.write(name_bytes)
            f.write(struct.pack('<II', layer['out_features'], layer['in_features']))
            layer_type = 1 if layer.get('is_raw', False) else 0
            f.write(struct.pack('<B', layer_type))     # 0=I2_S, 1=RAW_FP32
            f.write(struct.pack('<I', len(layer['data'])))
            f.write(layer['data'])

    file_size = os.path.getsize(bin_path)

    total_raw_bytes = sum(len(l['data']) for l in q_layers if l.get('is_raw', False))
    print("\n" + "=" * 70)
    print("COMPRESSION METRICS (BitNet I2_S)")
    print("=" * 70)
    print(f"  I2_S quantized layers:       {n_quantized}")
    print(f"  Raw FP32 layers:             {n_skipped}")
    print(f"  Original FP32 (quant only):  {total_fp32_bytes / 1e6:.1f} MB")
    print(f"  I2_S compressed:             {total_i2s_bytes / 1e6:.1f} MB")
    print(f"  Raw FP32 data:               {total_raw_bytes / 1e6:.1f} MB")
    print(f"  Binary file:                 {file_size / 1e6:.1f} MB")
    total_comp = total_i2s_bytes + total_raw_bytes
    print(f"  Compression ratio vs FP32:   {total_fp32_bytes / total_comp:.1f}x")
    print(f"  Bits per param (I2_S only):  {total_i2s_bytes * 8 / (total_fp32_bytes / 4):.2f}")
    print(f"  Quantization time:           {t_quant:.1f}s")

    meta = {
        'model': 'HuggingFaceTB/SmolLM2-135M',
        'format': 'I2_S', 'n_quantized': n_quantized, 'n_skipped': n_skipped,
        'original_fp32_mb': total_fp32_bytes / 1e6,
        'compressed_mb': total_i2s_bytes / 1e6,
        'compression_ratio': total_fp32_bytes / total_i2s_bytes,
        'bits_per_param': total_i2s_bytes * 8 / (total_fp32_bytes / 4),
        'quant_time_s': t_quant,
        'layer_errors': {l['name']: round(l['err'], 2) for l in q_layers},
    }
    meta_path = out_dir / "export_meta_bitnet.json"
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"\nMetadata saved to {meta_path}")
    print("\nDone!")

if __name__ == '__main__':
    main()
