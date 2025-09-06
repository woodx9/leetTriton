import torch
import triton
import triton.language as tl


@triton.jit
def matrix_copy_kernel(
    a_ptr, b_ptr, 
    M, N,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    a_ptrs = a_ptr + offs_m[:, None] * N + offs_n[None, :]
    b_ptrs = b_ptr + offs_m[:, None] * N + offs_n[None, :]

    a = tl.load(a_ptrs, mask=mask, other=0.0)
    tl.store(b_ptrs, a, mask=mask)



# a, b are tensors on the GPU
def solve(a: torch.Tensor, b: torch.Tensor, N: int):
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16

    M, N = a.shape
   
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
    matrix_copy_kernel[grid](
        a, b,
        M, N,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N
    )


# ---------------------------------------------------------------------------
# Tests (appended, original code above unchanged)
# ---------------------------------------------------------------------------
def test_matrix_copy():
    """Run several matrix copy scenarios in a single function."""
    if not torch.cuda.is_available():
        print("[SKIP] CUDA not available; tests skipped.")
        return

    device = torch.device("cuda")
    torch.manual_seed(0)

    def run_case(shape, desc):
        M, N = shape
        a = torch.randn(M, N, device=device, dtype=torch.float32)
        b = torch.empty_like(a)
        solve(a, b, N)
        if not torch.allclose(a, b):
            raise AssertionError(f"Matrix copy failed for {desc} shape={shape}. Max abs diff={(a-b).abs().max().item():.3e}")
        print(f"[OK] {desc} shape={shape}")

    # 1. Small square
    run_case((4, 4), "small square")
    # 2. Rectangular (more rows)
    run_case((5, 3), "rectangular tall")
    # 3. Rectangular (more cols)
    run_case((3, 7), "rectangular wide")
    # 4. Non-multiple of block sizes (prime-ish dims)
    run_case((17, 23), "non-multiple block")
    # 5. Larger but still quick
    run_case((64, 96), "moderate size")
    # 6. Edge-ish minimal (1x1)
    run_case((1, 1), "single element")

    print("All matrix copy tests passed.")


if __name__ == "__main__":
    test_matrix_copy()
