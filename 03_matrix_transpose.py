import time
import torch
import triton
import triton.language as tl

@triton.jit
def matrix_transpose_kernel(
    a_ptr, b_ptr,
    rows, cols,
    stride_ir, stride_ic,  
    stride_or, stride_oc,
    BLOCK: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK + tl.arange(0, BLOCK)
    offs_n = pid_n * BLOCK + tl.arange(0, BLOCK)

    a_ptrs  = a_ptr + offs_m[:, None] * stride_ir + offs_n[None, :] * stride_ic
    mask = (offs_m[:, None] < rows) & (offs_n[None, :] < cols)

    tile = tl.load(a_ptrs, mask=mask, other=0.0)

    tile_T = tl.trans(tile)
    
    mask_T = (offs_n[:, None] < cols) & (offs_m[None, :] < rows)

    b_ptrs = b_ptr + offs_n[:, None] * stride_or + offs_m[None, :] * stride_oc
    tl.store(b_ptrs, tile_T, mask=mask_T)

# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, rows: int, cols: int):
    stride_ir, stride_ic = cols, 1  
    stride_or, stride_oc = rows, 1
    

    BLOCK = 16
    grid = (triton.cdiv(rows, BLOCK), triton.cdiv(cols, BLOCK))
    matrix_transpose_kernel[grid](
        input, output,
        rows, cols,
        stride_ir, stride_ic,
        stride_or, stride_oc,
        BLOCK
    )
    
    return output

def triton_transpose(a: torch.Tensor):
    """用户层接口：只传一个 tensor，返回转置结果"""
    M, N = a.shape
    b = torch.empty((N, M), device=a.device, dtype=a.dtype)
    
    solve(a, b, M, N)
    return b


def simple_test_case():
    # 测试用例1：方阵
    a = torch.arange(16, dtype=torch.float32, device='cuda').reshape(4, 4)
    b = torch.empty((4, 4), dtype=torch.float32, device='cuda')
    solve(a, b, 4, 4)
    assert torch.allclose(b.cpu(), a.cpu().T), f"4x4 方阵转置失败\n原始:\n{a.cpu()}\n转置:\n{b.cpu()}"

    # 测试用例2：非方阵
    a = torch.arange(12, dtype=torch.float32, device='cuda').reshape(3, 4)
    b = torch.empty((4, 3), dtype=torch.float32, device='cuda')
    solve(a, b, 3, 4)
    assert torch.allclose(b.cpu(), a.cpu().T), f"3x4 非方阵转置失败\n原始:\n{a.cpu()}\n转置:\n{b.cpu()}"

    # 测试用例3：1xN
    a = torch.arange(5, dtype=torch.float32, device='cuda').reshape(1, 5)
    b = torch.empty((5, 1), dtype=torch.float32, device='cuda')
    solve(a, b, 1, 5)
    assert torch.allclose(b.cpu(), a.cpu().T), f"1x5 行向量转置失败\n原始:\n{a.cpu()}\n转置:\n{b.cpu()}"

    # 测试用例4：Nx1
    a = torch.arange(7, dtype=torch.float32, device='cuda').reshape(7, 1)
    b = torch.empty((1, 7), dtype=torch.float32, device='cuda')
    solve(a, b, 7, 1)
    assert torch.allclose(b.cpu(), a.cpu().T), f"7x1 列向量转置失败\n原始:\n{a.cpu()}\n转置:\n{b.cpu()}"

    # 测试用例5：空矩阵
    a = torch.empty((0, 3), dtype=torch.float32, device='cuda')
    b = torch.empty((3, 0), dtype=torch.float32, device='cuda')
    solve(a, b, 0, 3)
    assert b.numel() == 0, "空矩阵转置失败"

    print("所有矩阵转置测试通过！")


def benchmark():
    device = "cuda"
    dtype = torch.float32

    sizes = [
        (512, 512),
        (1024, 1024),
        (2048, 2048),
        (4096, 4096),
        (1024, 2048),   # 非方阵
        (2048, 1024),   # 非方阵
        (3000, 5000),   # 非方阵
    ]

    print(f"{'Shape':>15} | {'Torch (ms)':>12} | {'Triton (ms)':>12}")
    print("-" * 45)

    # 预热：避免首次包含 JIT 编译时间
    _ = triton_transpose(torch.randn((32, 64), device=device, dtype=dtype))

    for (m, n) in sizes:
        a = torch.randn((m, n), device=device, dtype=dtype)

        # Torch baseline
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(10):
            b1 = a.T.contiguous()
        torch.cuda.synchronize()
        torch_time = (time.time() - start) * 1000 / 10

        # Triton kernel
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(10):
            b2 = triton_transpose(a)   # Triton 版本
        torch.cuda.synchronize()
        triton_time = (time.time() - start) * 1000 / 10

        # 校验正确性
        assert torch.allclose(b1, b2)

        print(f"{str((m,n)):>15} | {torch_time:12.3f} | {triton_time:12.3f}")

if __name__ == "__main__":
    benchmark()