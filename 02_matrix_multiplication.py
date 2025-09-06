import triton
import triton.language as tl
import torch

@triton.jit
def matrix_multiplication_kernel_v1(
    a_ptr, b_ptr, c_ptr, 
    M, N, K,
    stride_am, stride_an, 
    stride_bn, stride_bk, 
    stride_cm, stride_ck
):
    pid_m = tl.program_id(0)  # M dimension
    pid_k = tl.program_id(1)  # K dimension

     # 初始化累加器
    accumulator = 0.0
    
    # 遍历N维度进行内积计算
    for n in range(0, N):
        # 计算A矩阵中a[pid_m, n]的位置
        a_offset = pid_m * stride_am + n * stride_an
        a_val = tl.load(a_ptr + a_offset)
        
        # 计算B矩阵中b[n, pid_k]的位置
        b_offset = n * stride_bn + pid_k * stride_bk
        b_val = tl.load(b_ptr + b_offset)
        
        # 累加乘积
        accumulator += a_val * b_val
    
    # 计算C矩阵中c[pid_m, pid_k]的位置并存储结果
    c_offset = pid_m * stride_cm + pid_k * stride_ck
    tl.store(c_ptr + c_offset, accumulator)

# a, b, c are tensors on the GPU
def solve_v1(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, M: int, N: int, K: int):
    stride_am, stride_an = N, 1 
    stride_bn, stride_bk = K, 1  
    stride_cm, stride_ck = K, 1
    
    grid = (M, K) 
    matrix_multiplication_kernel_v1[grid](
        a, b, c,
        M, N, K,
        stride_am, stride_an,
        stride_bn, stride_bk,
        stride_cm, stride_ck
    )

@triton.jit
def matrix_multiplication_kernel_v2(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_an,
    stride_bn, stride_bk,
    stride_cm, stride_ck,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)  # M dimension
    pid_k = tl.program_id(1)  # K dimension

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for n in range(0, N, BLOCK_K):
        offs_n = n + tl.arange(0, BLOCK_K)

        a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_n[None, :] * stride_an)
        b_ptrs = b_ptr + (offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk)

        # mask 避免越界访问
        a_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        b_mask = (offs_n[:, None] < N) & (offs_k[None, :] < K)

        # 读取子块数据（越界补 0）
        a_sub = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b_sub = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # forbidden allow_tf32 is important for accuracy
        acc += tl.dot(a_sub, b_sub, allow_tf32=False)

    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_k[None, :] * stride_ck)
    c_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
    tl.store(c_ptrs, acc, mask=c_mask)
   

# tile version
def solve_v2(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, M: int, N: int, K: int):
    stride_am, stride_an = N, 1
    stride_bn, stride_bk = K, 1
    stride_cm, stride_ck = K, 1
    

    BLOCK_M = 16
    BLOCK_N = 16
    BLOCK_K = 16

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(K, BLOCK_K))
    matrix_multiplication_kernel_v2[grid](
        a, b, c,
        M, N, K,
        stride_am, stride_an,
        stride_bn, stride_bk,
        stride_cm, stride_ck,
        BLOCK_M, BLOCK_N, BLOCK_K
    )


if __name__ == "__main__":
    # 测试用例1：随机矩阵
    solve = solve_v2

    M, N, K = 4, 5, 6
    a = torch.randn((M, N), device='cuda').to(torch.float32)
    b = torch.randn((N, K), device='cuda').to(torch.float32)

    # torch 默认用tf32，导致精度测试有问题
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    torch_output = a @ b
    triton_output = torch.empty((M, K), device='cuda').to(torch.float32)
    solve(a, b, triton_output, M, N, K)
    print("Test 1:", "✅" if torch.allclose(triton_output, torch_output, atol=1e-3, rtol=1e-3) else "❌")
    # 测试用例2：全1矩阵
    M, N, K = 3, 3, 3
    a = torch.ones((M, N), device='cuda')
    b = torch.ones((N, K), device='cuda')
    torch_output = a @ b
    triton_output = torch.empty((M, K), device='cuda')
    solve(a, b, triton_output, M, N, K)
    print("Test 2:", "✅" if torch.allclose(triton_output, torch_output) else "❌")

    # 测试用例3：单位矩阵
    M, N, K = 2, 2, 2
    a = torch.eye(M, N, device='cuda')
    b = torch.eye(N, K, device='cuda')
    torch_output = a @ b
    triton_output = torch.empty((M, K), device='cuda')
    solve(a, b, triton_output, M, N, K)
    print("Test 3:", "✅" if torch.allclose(triton_output, torch_output) else "❌")
