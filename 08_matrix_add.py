import torch
import triton
import triton.language as tl


@triton.jit
def matrix_copy_kernel(
    a_ptr, b_ptr, c_ptr,
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
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]

    a = tl.load(a_ptrs, mask=mask, other=0.0)
    b = tl.load(b_ptrs, mask=mask, other=0.0)

    c = a + b
    tl.store(c_ptrs, c, mask=mask)



# a, b are tensors on the GPU
def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, N: int):
    assert a.shape == b.shape
    assert b.shape == c.shape

    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16

    M, N = a.shape
   
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
    matrix_copy_kernel[grid](
        a, b, c,
        M, N,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N
    )


# ---------------------------------------------------------------------------
# Tests (added below without altering original implementation above)
# ---------------------------------------------------------------------------
def test_matrix_add():
    """Run several matrix add scenarios in one function.

    Expected semantics (what we assert): c = a + b (element-wise).
    NOTE: If this test fails, inspect the kernel's loads/stores order.
    """
    if not torch.cuda.is_available():
        print("[SKIP] CUDA not available; tests skipped.")
        return

    device = torch.device("cuda")
    torch.manual_seed(0)

    def run_case(shape, desc, dtype=torch.float32):
        M, N = shape
        a = torch.randn(M, N, device=device, dtype=dtype)
        b = torch.randn(M, N, device=device, dtype=dtype)
        # Output tensor initialized (could be any values)
        c = torch.empty_like(a)
        solve(a, b, c, N)
        expected = a + b
        if not torch.allclose(c, expected, atol=1e-5, rtol=1e-5):
            max_diff = (c - expected).abs().max().item()
            raise AssertionError(
                f"Matrix add failed for {desc} shape={shape} dtype={dtype}. Max abs diff={max_diff:.3e}"
            )
        print(f"[OK] {desc} shape={shape} dtype={dtype}")

    # 1. Small square
    run_case((4, 4), "small square")
    # 2. Rectangular tall
    run_case((5, 3), "rectangular tall")
    # 3. Rectangular wide
    run_case((3, 7), "rectangular wide")
    # 4. Non-multiple of block sizes
    run_case((17, 23), "non-multiple block")
    # 5. Larger moderate
    run_case((64, 96), "moderate size")
    # 6. Single element
    run_case((1, 1), "single element")
    # 7. Different dtype (float16) - small to keep numerical diff reasonable
    run_case((8, 8), "float16 small", dtype=torch.float16)

    print("All matrix add tests passed.")


if __name__ == "__main__":
    test_matrix_add()