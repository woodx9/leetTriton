import torch
import triton
import triton.language as tl

@triton.jit
def vector_add_kernel(a_ptr, b_ptr, c_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    c = a + b
    tl.store(c_ptr + offsets, c, mask=mask)

def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, N: int):
    BLOCK_SIZE = 16
    grid = (triton.cdiv(N, BLOCK_SIZE), )

    vector_add_kernel[grid](a, b, c, N, BLOCK_SIZE)

if __name__ == "__main__":
    N = 20
    a = torch.randn(N, device='cuda')
    b = torch.randn(N, device='cuda')
    torch_output = a + b
    triton_output = torch.empty_like(a)
    solve(a, b , triton_output, N)
    if torch.allclose(triton_output, torch_output):
        print("✅ Triton and Torch match")
    else:
        print("❌ Triton and Torch differ")