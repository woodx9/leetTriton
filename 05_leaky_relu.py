import torch
import triton
import triton.language as tl

@triton.jit
def leaky_relu_kernel(
    a_ptr,
    b_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid_m = tl.program_id(axis=0)
    alpha: tl.constexpr = 0.01

    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs_m < n_elements

    a = tl.load(a_ptr + offs_m, mask=mask, other=0.0)
    # ReLU: max(a, 0)
    b = tl.where(a > 0, a, a * alpha)
    tl.store(b_ptr + offs_m, b, mask=mask)

# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    leaky_relu_kernel[grid](
        input,
        output,
        N,
        BLOCK_SIZE
    )


def test_leaky_relu():
    """Simple correctness test comparing Triton kernel vs PyTorch reference."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this Triton test.")

    device = torch.device("cuda")
    torch.manual_seed(0)

    # Pick a size that's NOT a multiple of BLOCK_SIZE to exercise the mask path
    BLOCK_SIZE = 1024
    N = BLOCK_SIZE * 5 + 137  # arbitrary non-multiple

    x = torch.randn(N, device=device, dtype=torch.float32)
    y = torch.empty_like(x)

    # Run Triton implementation
    solve(x, y, N)

    # PyTorch reference (negative_slope = 0.01)
    ref = torch.nn.functional.leaky_relu(x, negative_slope=0.01)

    max_diff = (y - ref).abs().max().item()
    print(f"Max abs diff: {max_diff:.3e}")
    tol = 1e-6
    if max_diff > tol:
        # Provide a small diagnostic sample
        mismatch_idx = (y - ref).abs().argmax().item()
        print("Mismatch sample:")
        print("x=", x[mismatch_idx].item(), " triton=", y[mismatch_idx].item(), " ref=", ref[mismatch_idx].item())
        raise AssertionError(f"Test failed: max diff {max_diff} > {tol}")
    print("LeakyReLU test passed ✅")


if __name__ == "__main__":
    test_leaky_relu()