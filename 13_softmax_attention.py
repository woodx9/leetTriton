import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import time
import numpy as np

## 假设 Q: [M, d], K: [N, d], V: [N, d]
# # 1. 计算相似度矩阵 QK^T / sqrt(d)
# scores = Q @ K.T / (d ** 0.5)  # [M, N]

# # 2. 对每行做 softmax
# weights = F.softmax(scores, dim=1)  # [M, N]

# # 3. 用权重加权 V
# output = weights @ V  # [M, d]


@triton.jit
def softmax_attention_kernel(
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    M, N, D,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr
):
    row = tl.program_id(0)

    offs_d = tl.arange(0, BLOCK_SIZE_D)
    mask_q = offs_d < D
    q = tl.load(Q_ptr + row * D + offs_d, mask=mask_q, other=0.0)
    q = tl.reshape(q, [1, BLOCK_SIZE_D])

    out = tl.zeros([BLOCK_SIZE_D], dtype=tl.float32)
    
    global_max = -float("inf")
    total_exp_sum = 0.0
    scale = tl.sqrt(tl.cast(D, tl.float32))

    # 全局max
    for j in range(0, N, BLOCK_SIZE_N):
        offs_n = j + tl.arange(0, BLOCK_SIZE_N)
        mask_kv = (offs_n[:, None] < N) & (offs_d[None, :] < D)
        
        k = tl.load(K_ptr + offs_n[:, None] * D + offs_d[None, :], mask=mask_kv, other=0.0)
        s = tl.sum(k * q, axis=1) / scale
        
        mask_n = offs_n < N
        s = tl.where(mask_n, s, -float("inf"))
        
        local_max = tl.max(s, axis=0)
        global_max = tl.maximum(global_max, local_max)

    for j in range(0, N, BLOCK_SIZE_N):
        offs_n = j + tl.arange(0, BLOCK_SIZE_N)
        mask_kv = (offs_n[:, None] < N) & (offs_d[None, :] < D)
        
        k = tl.load(K_ptr + offs_n[:, None] * D + offs_d[None, :], mask=mask_kv, other=0.0)
        v = tl.load(V_ptr + offs_n[:, None] * D + offs_d[None, :], mask=mask_kv, other=0.0)
        
        s = tl.sum(k * q, axis=1) / scale
        mask_n = offs_n < N
        s = tl.where(mask_n, s, -float("inf"))
        
        exp_s = tl.exp(s - global_max)
        total_exp_sum += tl.sum(exp_s, axis=0)
        
        w = tl.reshape(exp_s, [BLOCK_SIZE_N, 1])
        out += tl.sum(w * v, axis=0)

    # softmax里 / total_exp_sum
    out = out / total_exp_sum
    tl.store(Out_ptr + row * D + offs_d, out, mask=mask_q)



def solve(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, output: torch.Tensor, M: int, N: int, d: int):
    grid = (M,)  
    softmax_attention_kernel[grid](
        Q, K, V, output,
        M, N, d,
        BLOCK_SIZE_M=1,
        BLOCK_SIZE_N=min(128, N),    
        BLOCK_SIZE_D=d                
    )

def torch_softmax_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
    """PyTorch实现的softmax attention"""
    d = Q.shape[-1]
    # 1. 计算相似度矩阵 QK^T / sqrt(d)
    scores = Q @ K.T / (d ** 0.5)  # [M, N]
    # 2. 对每行做 softmax
    weights = F.softmax(scores, dim=1)  # [M, N]
    # 3. 用权重加权 V
    output = weights @ V  # [M, d]
    return output


def benchmark_softmax_attention():
    """测试Triton和PyTorch softmax attention的性能对比"""
    print("\n" + "=" * 90)
    print("Softmax Attention Performance Benchmark")
    print("=" * 90)
    
    # 表头
    header = f"{'M':<6} {'N':<6} {'d':<6} {'Triton(ms)':<12} {'PyTorch(ms)':<13} {'Speedup':<8} {'Max Error':<12} {'Status':<8}"
    print(header)
    print("-" * 90)
    
    # 测试不同的矩阵尺寸
    test_cases = [
        (128, 128, 1024),
        (128, 128, 64),
        (256, 256, 64),
        (512, 512, 64),
        (128, 128, 128),
        (1024, 1024, 64),
        (2048, 2048, 64),
        (1024, 1024, 128),
        (2048, 2048, 128),
        (1024, 4096, 64),
        (2048, 8192, 64),
        (1024, 1024, 256),
        (2048, 2048, 256),
        (4096, 4096, 128),
        (1024, 1024, 128),
        (2048, 2048, 128)
    ]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        print("Warning: CUDA not available, running on CPU. Triton requires CUDA.")
        return
    
    for M, N, d in test_cases:
        # 生成随机测试数据
        Q = torch.randn(M, d, device=device, dtype=torch.float32)
        K = torch.randn(N, d, device=device, dtype=torch.float32)
        V = torch.randn(N, d, device=device, dtype=torch.float32)
        
        # 预热GPU
        for _ in range(5):
            _ = torch_softmax_attention(Q, K, V)
            triton_output = torch.zeros(M, d, device=device, dtype=torch.float32)
            solve(Q, K, V, triton_output, M, N, d)
        
        torch.cuda.synchronize()
        
        # 测试PyTorch实现
        torch_times = []
        for _ in range(50):
            start = time.perf_counter()
            torch_output = torch_softmax_attention(Q, K, V)
            torch.cuda.synchronize()
            end = time.perf_counter()
            torch_times.append((end - start) * 1000)
        
        # 测试Triton实现
        triton_times = []
        for _ in range(50):
            triton_output = torch.zeros(M, d, device=device, dtype=torch.float32)
            start = time.perf_counter()
            solve(Q, K, V, triton_output, M, N, d)
            torch.cuda.synchronize()
            end = time.perf_counter()
            triton_times.append((end - start) * 1000)
        
        # 计算平均时间
        torch_avg = np.mean(torch_times)
        triton_avg = np.mean(triton_times)
        speedup = torch_avg / triton_avg
        
        # 验证结果正确性
        torch_output = torch_softmax_attention(Q, K, V)
        triton_output = torch.zeros(M, d, device=device, dtype=torch.float32)
        solve(Q, K, V, triton_output, M, N, d)
        
        max_diff = torch.max(torch.abs(torch_output - triton_output)).item()
        mean_diff = torch.mean(torch.abs(torch_output - triton_output)).item()
        
        # 判断精度状态
        if max_diff < 1e-3:
            status = "PASS"
        elif max_diff < 1e-2:
            status = "WARN"
        else:
            status = "FAIL"
        
        # 格式化输出
        row = f"{M:<6} {N:<6} {d:<6} {triton_avg:<12.3f} {torch_avg:<13.3f} {speedup:<8.2f}x {max_diff:<12.2e} {status:<8}"
        print(row)
    
    print("-" * 90)
    print("Benchmark completed!")
    print("Status: PASS(<1e-3), WARN(<1e-2), FAIL(>=1e-2)")
    print("=" * 90)


if __name__ == "__main__":
    benchmark_softmax_attention()
