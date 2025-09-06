import torch
import triton
import triton.language as tl

@triton.jit
def fnv1a_hash(x):
    FNV_PRIME = 16777619
    OFFSET_BASIS = 2166136261
    
    hash_val = tl.full(x.shape, OFFSET_BASIS, tl.uint32)
    
    for byte_pos in range(4):
        byte = (x >> (byte_pos * 8)) & 0xFF
        hash_val = (hash_val ^ byte) * FNV_PRIME
    
    return hash_val

@triton.jit
def fnv1a_hash_kernel(
    a_ptr,
    b_ptr,
    n_elements,
    n_rounds,
    BLOCK_SIZE: tl.constexpr
):
    pid_m = tl.program_id(axis=0)

    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs_m < n_elements

    a = tl.load(a_ptr + offs_m, mask=mask, other=0).to(tl.uint32)

    for _ in range(n_rounds):
        a = fnv1a_hash(a)

    b = a
    tl.store(b_ptr + offs_m, b, mask=mask)

# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int, R: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    fnv1a_hash_kernel[grid](
        input,
        output,
        N,
        R,
        BLOCK_SIZE
    )


def _assert_example(numbers, R, expected):
    if not torch.cuda.is_available():
        print("CUDA 不可用，跳过测试 (需要 GPU 才能运行 Triton kernel)")
        return
    device = 'cuda'
    inp = torch.tensor(numbers, dtype=torch.int32, device=device)
    out = torch.empty_like(inp)
    solve(inp, out, inp.numel(), R)
    # 转成无符号 32 位整数对比
    got = [(int(v.item()) & 0xFFFFFFFF) for v in out.cpu()]
    if got != expected:
        raise AssertionError(f"输入 {numbers}, R={R} 期望 {expected} 实际 {got}")


def test_rainbow_table():
    # 示例 1
    _assert_example([123, 456, 789], 2, [1636807824, 1273011621, 2193987222])
    # 示例 2
    _assert_example([0, 1, 2147483647], 3, [96754810, 3571711400, 2006156166])
    print("Rainbow table 单测通过 ✅")


if __name__ == "__main__":
    test_rainbow_table()