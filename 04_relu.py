import torch
import triton
import triton.language as tl

@triton.jit
def relu_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid_m = tl.program_id(axis=0)

    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs_m < n_elements

    a = tl.load(a_ptr + offs_m, mask=mask, other=0.0)
    # ReLU: max(a, 0)
    b = tl.maximum(a, 0)
    tl.store(b_ptr + offs_m, b, mask=mask)



    
# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    relu_kernel[grid](input, output, N, BLOCK_SIZE)


def relu(input: torch.Tensor):
    """Convenience wrapper returning a new tensor."""
    out = torch.empty_like(input)
    solve(input, out, input.numel())
    return out


def _run_tests():
    device = 'cuda'
    dtype = torch.float32

    def check(x):
        y_ref = torch.relu(x)
        y_out = relu(x)
        assert torch.allclose(y_ref, y_out), f"Mismatch\nref={y_ref}\nout={y_out}"

    # 1. Mixed positive/negative
    x = torch.tensor([-3.0, -0.1, 0.0, 0.2, 5.5], device=device, dtype=dtype)
    check(x)

    # 2. All negative
    x = torch.linspace(-5, -1, 17, device=device, dtype=dtype)
    check(x)

    # 3. All positive
    x = torch.linspace(0.1, 9.9, 23, device=device, dtype=dtype)
    check(x)

    # 4. Random large (not multiple of BLOCK_SIZE)
    x = torch.randn(10_000 + 123, device=device, dtype=dtype)
    check(x)

    # 5. Empty
    x = torch.empty(0, device=device, dtype=dtype)
    y = relu(x)
    assert y.numel() == 0

    print("ReLU tests passed.")


if __name__ == "__main__":
    _run_tests()
