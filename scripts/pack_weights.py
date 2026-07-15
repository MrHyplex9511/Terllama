"""
I2_S pack/unpack. BitNet ternary. 4 vals/byte.
"""
import torch
import struct

def pack_i2s(codes, qk=128):
    """Pack ternary codes → I2_S. 4 vals/byte.

    Args:
        codes: int8 tensor {-1, 0, +1}
        qk: block size (default 128)

    Returns:
        packed: bytes
        n_elements: element count

    Mapping: {-1,0,+1}→{0,1,2}→2-bit codes
    Byte (MSB→LSB): [elem0(2b)][elem1(2b)][elem2(2b)][elem3(2b)]
    """
    # {-1,0,+1} -> {0,1,2} encoding
    mapped = codes.to(torch.int8) + 1
    flat = mapped.flatten().to(torch.uint8)
    n = flat.numel()

    # Pad to QK boundary
    pad = (qk - n % qk) % qk
    if pad > 0:
        flat = torch.cat([flat, torch.zeros(pad, dtype=torch.uint8)])

    packed = bytearray()
    for i in range(0, flat.numel(), 4):
        byte = ((flat[i]   & 0x03) << 6) | \
               ((flat[i+1] & 0x03) << 4) | \
               ((flat[i+2] & 0x03) << 2) | \
               (flat[i+3] & 0x03)
        packed.append(byte)

    return bytes(packed), n


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
    out_f, in_f = codes.shape
    n_blocks = scales.shape[1]
    codes_per_block = qk // 4  # 32 bytes per block
    buf = bytearray()
    for row in range(out_f):
        # Row codes → list
        row_codes = codes[row].tolist()  # len = in_f
        for b in range(n_blocks):
            block_start = b * qk
            block_end = min(block_start + qk, in_f)
            block_size = block_end - block_start
            # Pack qk vals
            byte_buf = bytearray(codes_per_block)
            for j in range(0, qk, 4):
                byte_val = 0
                for k in range(4):
                    idx = block_start + j + k
                    if idx < in_f:
                        # {-1,0,+1} -> {0,1,2}
                        val = row_codes[idx] + 1
                    else:
                        val = 0  # zero-pad
                    byte_val |= (val & 0x03) << (6 - k * 2)
                byte_buf[j // 4] = byte_val
            buf.extend(byte_buf)
            # Append row's block scale
            buf.extend(struct.pack('<f', scales[row, b].item()))
    return bytes(buf)


def unpack_i2s(packed_bytes, n_elements, qk=128):
    """Unpack I2_S → int8 tensor {-1,0,+1}.

    Args:
        packed_bytes: I2_S packed data
        n_elements: original element count
        qk: block size

    Returns:
        int8 tensor {-1, 0, +1}
    """
    n_bytes = (n_elements + 3) // 4  # 4 values per byte
    codes = torch.zeros(n_elements, dtype=torch.int8)
    for i in range(min(n_bytes, len(packed_bytes))):
        byte = packed_bytes[i]
        for j in range(4):
            idx = i * 4 + j
            if idx >= n_elements:
                break
            val = (byte >> (6 - j * 2)) & 0x03
            # {0,1,2} -> {-1,0,+1}
            codes[idx] = val - 1
    return codes
