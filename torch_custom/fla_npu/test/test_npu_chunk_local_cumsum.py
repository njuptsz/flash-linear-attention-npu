import math
import os
from typing import List, Optional, Tuple

import torch
import torch_npu
import fla_npu


torch.npu.config.allow_internal_format = False
torch.npu.set_compile_mode(jit_compile=False)
torch.npu.set_device(int(os.environ.get("TEST_DEVICE_ID", 0)))


def _next_power_of_two(value: int) -> int:
    value = max(value, 1)
    return 1 << (value - 1).bit_length()


def _block_t(num_heads: int, chunk_size: int) -> int:
    return _next_power_of_two((1 << 17) // (num_heads * chunk_size))


def prepare_chunk_indices(cu_seqlens: torch.Tensor, block_t: int) -> torch.Tensor:
    rows = []
    for seq_idx, (start, end) in enumerate(zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist())):
        num_blocks = math.ceil((end - start) / block_t)
        for block_idx in range(num_blocks):
            rows.append((seq_idx, block_idx))
    return torch.tensor(rows, dtype=torch.long)


def reference_impl(
    g: torch.Tensor,
    chunk_size: int,
    reverse: bool,
    scale: float,
    cu_seqlens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    out = torch.empty_like(g, dtype=torch.float32)
    if cu_seqlens is None:
        ranges = [(b, 0, g.size(1)) for b in range(g.size(0))]
    else:
        ranges = [(0, start, end) for start, end in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist())]

    for batch, seq_start, seq_end in ranges:
        seq_len = seq_end - seq_start
        for chunk_start in range(0, seq_len, chunk_size):
            start = seq_start + chunk_start
            end = min(start + chunk_size, seq_end)
            segment = g[batch, start:end, :].to(torch.float32)
            if reverse:
                value = torch.flip(torch.cumsum(torch.flip(segment, dims=[0]), dim=0), dims=[0])
            else:
                value = torch.cumsum(segment, dim=0)
            out[batch, start:end, :] = value * scale
    return out


def run_case(
    name: str,
    shape: Tuple[int, int, int],
    chunk_size: int,
    reverse: bool = False,
    scale: float = 1.0,
    cu_seqlens_values: Optional[List[int]] = None,
) -> None:
    torch.manual_seed(sum(ord(ch) for ch in name))
    g_cpu = torch.randn(shape, dtype=torch.float32)
    g_npu = g_cpu.npu()

    cu_seqlens_npu = None
    chunk_indices_npu = None
    cu_seqlens_cpu = None
    if cu_seqlens_values is not None:
        cu_seqlens_cpu = torch.tensor(cu_seqlens_values, dtype=torch.long)
        block_t = _block_t(shape[2], chunk_size)
        chunk_indices_cpu = prepare_chunk_indices(cu_seqlens_cpu, block_t)
        cu_seqlens_npu = cu_seqlens_cpu.npu()
        chunk_indices_npu = chunk_indices_cpu.npu()

    actual = torch.ops.npu.npu_chunk_local_cumsum(
        g_npu,
        chunk_size,
        cu_seqlens=cu_seqlens_npu,
        chunk_indices_out=chunk_indices_npu,
        reverse=reverse,
        scale=scale,
        head_first=False,
        output_dtype="float32",
    ).cpu()
    expected = reference_impl(g_cpu, chunk_size, reverse, scale, cu_seqlens_cpu)

    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
    print(f"[PASS] {name}: shape={shape}, chunk_size={chunk_size}, reverse={reverse}, scale={scale}")


if __name__ == "__main__":
    run_case("fixed_bsnd_like_forward", (9, 128, 2), chunk_size=64)
    run_case("fixed_bsnd_like_reverse_scale", (9, 128, 2), chunk_size=64, reverse=True, scale=0.25)
    run_case("varlen_tnd_like_forward", (1, 3580, 8), chunk_size=64, cu_seqlens_values=[0, 3580])
    run_case(
        "varlen_tnd_like_reverse_scale",
        (1, 3580, 8),
        chunk_size=64,
        reverse=True,
        scale=-0.5,
        cu_seqlens_values=[0, 1024, 2048, 3580],
    )
