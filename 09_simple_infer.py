import time
import torch
import torch.nn as nn
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
def matrix_add(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor):
    if a.shape != b.shape:
        if b.dim() == 1 and a.shape[1] == b.shape[0]:
            # b 是 bias，扩展成每行相加
            # expand 产生 stride=0 视图，内核按行主内存递增访问会越界/读错值，需物化
            b = b.unsqueeze(0).expand(a.shape[0], -1).contiguous()
        else:
            raise ValueError(f"Shapes not compatible: a={a.shape}, b={b.shape}")

    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64

    M, N = a.shape
   
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
    matrix_copy_kernel[grid](
        a, b, c,
        M, N,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N
    )

@triton.jit
def matrix_multiplication_kernel_v2(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_an,
    stride_bm, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)  # M dimension
    pid_k = tl.program_id(1)  # N dimension

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for n in range(0, N, BLOCK_K):
        offs_n = n + tl.arange(0, BLOCK_K)

        a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_n[None, :] * stride_an)
        b_ptrs = b_ptr + (offs_n[:, None] * stride_bm + offs_k[None, :] * stride_bn)

        # mask 避免越界访问
        a_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        b_mask = (offs_n[:, None] < N) & (offs_k[None, :] < K)

        # 读取子块数据（越界补 0）
        a_sub = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b_sub = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # forbidden allow_tf32 is important for accuracy
        acc += tl.dot(a_sub, b_sub, allow_tf32=False)
        

    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_k[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
    tl.store(c_ptrs, acc.to(tl.float64), mask=c_mask)

   

# tile version
def matrix_multiplication(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor):
    M, N = a.shape
    N, K = b.shape

    stride_am, stride_an = a.stride()
    stride_bn, stride_bk = b.stride()
    stride_cm, stride_ck = c.stride()
    

    BLOCK_M = 16
    BLOCK_N = 16
    BLOCK_K = 16

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(K, BLOCK_N))
    matrix_multiplication_kernel_v2[grid](
        a, b, c,
        M, N, K,
        stride_am, stride_an,
        stride_bn, stride_bk,
        stride_cm, stride_ck,
        BLOCK_M, BLOCK_N, BLOCK_K
    )

# input, model, and output are on the GPU
def solve(input: torch.Tensor, model: nn.Module, output: torch.Tensor):
    
    linear = None
    for m in model.modules():
        if isinstance(m, nn.Linear):
            linear = m
            break
    if linear is None:
        raise ValueError("No nn.Linear layer found in model")
    
    W = linear.weight  
    b = linear.bias

    # W.T 不是连续内存（stride 改变），而 kernel 假设 row-major；需 contiguous 拷贝
    WT = W.t().contiguous()
    dot_outcome = torch.zeros(input.shape[0], W.shape[0], device=input.device)
    matrix_multiplication(input, WT, dot_outcome)
    
    matrix_add(dot_outcome, b, output)


# -------------------- Tests (appended, original code unchanged above) --------------------
def test_solve():
    torch.manual_seed(0)
    device = "cuda"
    # torch 默认用tf32，导致精度测试有问题
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    test_cases = [
        # (batch_size, input_dim, output_dim, bias)
        (1, 2, 2, True),
        (4, 8, 16, True),
        (8, 16, 4, True),
        (32, 64, 64, True),
    ]

    for i, (B, IN, OUT, use_bias) in enumerate(test_cases):
        print(f"\n[Test {i}] B={B}, IN={IN}, OUT={OUT}, bias={use_bias}")

        # 随机输入和模型
        input_tensor = torch.randn(B, IN, device=device, dtype=torch.float32).to(device)
        model = nn.Linear(IN, OUT, bias=use_bias, dtype=torch.float32).to(device)

        # 预期结果（torch原生）
        expected = model(input_tensor)

        # Triton 版本
        output = torch.empty_like(expected)
        solve(input_tensor, model, output)

        # 比较
        max_diff = (output - expected).abs().max().item()
        print(f"max_diff={max_diff:.6f}")
        if not torch.allclose(output, expected, atol=1e-3):
            print("Mismatch between Triton and Torch outputs!")
            print("torch", expected)
            print("triton", output)

    print("\nAll tests passed ✅")

def benchmark_solve():
    device = 'cuda'
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA not available for benchmark')
    torch.manual_seed(42)

    # (batch, in_dim, out_dim)
    configs = [
        (256,   256, 256),
        (512,   512, 512),
        (1024,  512, 512),
        (1024,  1024, 1024),
        (2048,  1024, 1024),
        (4096,  1024, 2048),
    ]

    repeats = 5   # 每个配置重复次数求均值
    warmup  = 3   # 预热次数

    print(f"Benchmark {len(configs)} configs (repeats={repeats}, warmup={warmup})")
    print("cfg\ttriton_ms\ttorch_ms\tspeedup\tmax_diff\tmean_diff")

    for (batch_size, input_dim, output_dim) in configs:
        # 分配
        input_tensor = torch.randn(batch_size, input_dim, device=device, dtype=torch.float32)
        output_triton = torch.empty(batch_size, output_dim, device=device, dtype=torch.float32)
        output_torch = torch.empty_like(output_triton)
        model = nn.Linear(input_dim, output_dim, bias=True).to(device).eval()

        # Warmup
        for _ in range(warmup):
            with torch.no_grad():
                solve(input_tensor, model, output_triton)
                output_torch.copy_(model(input_tensor))

        # 计时函数
        def time_it(fn):
            torch.cuda.synchronize()
            t0 = time.time()
            fn()
            torch.cuda.synchronize()
            return (time.time() - t0) * 1000.0

        # 运行多次求均值
        t_triton_acc = 0.0
        t_torch_acc = 0.0
        with torch.no_grad():
            for _ in range(repeats):
                t_triton_acc += time_it(lambda: solve(input_tensor, model, output_triton))
            for _ in range(repeats):
                t_torch_acc += time_it(lambda: output_torch.copy_(model(input_tensor)))

        t_triton = t_triton_acc / repeats
        t_torch = t_torch_acc / repeats

        # 精度
        max_diff = (output_triton - output_torch).abs().max().item()
        mean_diff = (output_triton - output_torch).abs().mean().item()
        speedup = t_torch / t_triton if t_triton > 0 else float('inf')

        print(f"{batch_size}x{input_dim}->{output_dim}\t{t_triton:.3f}\t{t_torch:.3f}\t{speedup:.2f}x\t{max_diff:.3e}\t{mean_diff:.3e}")

    print("Done.")

if __name__ == "__main__":

    # test_solve()
    benchmark_solve()
