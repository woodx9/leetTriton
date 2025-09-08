import torch
import triton
import triton.language as tl
import time
import torch.nn as nn
import numpy as np

# 简单处理，复用softmax attention
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

# Q, K, V, output are tensors on the GPU
# Q, K, V [B, N, d_model]
# d_model // h = d_head
def solve(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, output: torch.Tensor, N: int, d_model: int, h: int, num_kv_heads: int):
    """
    GQA implementation.
    Q: [B, N, d_model]
    K: [B, N, d_model]
    V: [B, N, d_model]
    output: [B, N, d_model]
    N: sequence length
    d_model: model dim
    h: num query heads
    num_kv_heads: num key/value heads (<= h)
    """

    B = Q.shape[0]
    assert d_model % h == 0, "d_model must be divisible by h"
    assert h % num_kv_heads == 0, "num_q_heads must be divisible by num_kv_heads"
     
    d_head = d_model // h
    heads_per_group = h // num_kv_heads
    M = N   # query_len == key_len


    # reshape to multi-head form [B, h, N, d_head]
    Qh = Q.view(B, N, h, d_head).transpose(1, 2).contiguous()  # [B, h, N, d_head]

    # [B, num_kv_heads, N, d_head * heads_per_group]
    Kh = K.view(B, N, num_kv_heads, d_head).transpose(1, 2).contiguous()  # [B, num_kv_heads, N, d_head]
    Vh = V.view(B, N, num_kv_heads, d_head).transpose(1, 2).contiguous()  # [B, num_kv_heads, N, d_head]
    
    Kh = Kh.repeat_interleave(heads_per_group, dim=1)
    Vh = Vh.repeat_interleave(heads_per_group, dim=1)
    
    Oh = output.view(B, N, h, d_head).transpose(1, 2).contiguous()

    # flatten batch*head
    Qf = Qh.reshape(B * h, N, d_head)
    Kf = Kh.reshape(B * h, N, d_head)
    Vf = Vh.reshape(B * h, N, d_head)
    Of = Oh.reshape(B * h, N, d_head)


    for batch_head_idx in range(B * h):
        grid = (M,)  # 每次只处理一个(batch,head)的M个查询
        softmax_attention_kernel[grid](
            Qf[batch_head_idx], Kf[batch_head_idx], Vf[batch_head_idx], Of[batch_head_idx],
            M, N, d_head,
            BLOCK_SIZE_M=1,
            BLOCK_SIZE_N=triton.next_power_of_2(min(128, N)),
            BLOCK_SIZE_D=triton.next_power_of_2(d_head),
        )


    # reshape back to [B, N, d_model]
    output.copy_(Oh.transpose(1, 2).reshape(B, N, d_model))




def torch_gqa(Q, K, V, num_kv_heads, h):
    """
    PyTorch reference implementation of GQA
    """
    B, N, d_model = Q.shape
    d_head = d_model // h
    heads_per_group = h // num_kv_heads
    kv_d_model = num_kv_heads * d_head
    
    # Reshape to multi-head
    Q = Q.view(B, N, h, d_head).transpose(1, 2)  # [B, h, N, d_head]
    K = K.view(B, N, num_kv_heads, d_head).transpose(1, 2)  # [B, num_kv_heads, N, d_head]
    V = V.view(B, N, num_kv_heads, d_head).transpose(1, 2)  # [B, num_kv_heads, N, d_head]
    
    # 扩展KV heads
    K = K.repeat_interleave(heads_per_group, dim=1)  # [B, h, N, d_head]
    V = V.repeat_interleave(heads_per_group, dim=1)  # [B, h, N, d_head]
    
    # Scaled dot-product attention
    scale = 1.0 / (d_head ** 0.5)
    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale  # [B, h, N, N]
    attn_weights = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn_weights, V)  # [B, h, N, d_head]
    
    # Reshape back
    out = out.transpose(1, 2).contiguous().view(B, N, d_model)
    return out

def benchmark_gqa():
    """
    Comprehensive benchmark comparing Triton GQA vs PyTorch GQA
    """
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    device = torch.cuda.current_device()
    print(f"Running benchmark on device: {device}")
    print("=" * 80)
    
    # 测试配置
    test_configs = [
        # (B, N, d_model, h, num_kv_heads)
        (1, 512, 512, 8, 8),      # Standard attention
        (1, 512, 512, 8, 4),      # GQA 2:1
        (1, 512, 512, 8, 2),      # GQA 4:1
        (1, 512, 512, 8, 1),      # MQA (extreme case)
        
        (2, 1024, 768, 12, 12),   # BERT-like
        (2, 1024, 768, 12, 6),    # BERT-like GQA
        (2, 1024, 768, 12, 4),    # BERT-like GQA
        
        (1, 2048, 1024, 16, 16),  # Larger model
        (1, 2048, 1024, 16, 8),   # Larger model GQA
        (1, 2048, 1024, 16, 4),   # Larger model GQA
    ]
    
    print(f"{'Config':<30} {'Triton (ms)':<12} {'PyTorch (ms)':<14} {'Speedup':<10} {'Max Diff':<12} {'Correct':<8}")
    print("-" * 90)
    
    for B, N, d_model, h, num_kv_heads in test_configs:
        try:
            d_head = d_model // h
            kv_d_model = num_kv_heads * d_head
            
            # 生成测试数据
            Q = torch.randn(B, N, d_model, device='cuda', dtype=torch.float32)
            K = torch.randn(B, N, kv_d_model, device='cuda', dtype=torch.float32)
            V = torch.randn(B, N, kv_d_model, device='cuda', dtype=torch.float32)
            
            # Triton版本
            output_triton = torch.zeros_like(Q)
            
            # PyTorch版本 
            Q_torch = Q.clone()
            K_torch = K.clone()
            V_torch = V.clone()
            
            # 正确性检查
            solve(Q.clone(), K.clone(), V.clone(), output_triton, N, d_model, h, num_kv_heads)
            output_torch = torch_gqa(Q_torch, K_torch, V_torch, num_kv_heads, h)
            
            # 检查数值正确性
            max_diff = torch.max(torch.abs(output_triton - output_torch)).item()
            is_correct = max_diff < 1e-3
            
            # 性能测试
            warmup_runs = 5
            test_runs = 20
            
            # Warmup
            for _ in range(warmup_runs):
                solve(Q.clone(), K.clone(), V.clone(), output_triton, N, d_model, h, num_kv_heads)
                torch_gqa(Q_torch.clone(), K_torch.clone(), V_torch.clone(), num_kv_heads, h)
            
            torch.cuda.synchronize()
            
            # Triton timing
            start_time = time.time()
            for _ in range(test_runs):
                solve(Q.clone(), K.clone(), V.clone(), output_triton, N, d_model, h, num_kv_heads)
            torch.cuda.synchronize()
            triton_time = (time.time() - start_time) * 1000 / test_runs
            
            # PyTorch timing
            start_time = time.time()
            for _ in range(test_runs):
                output_torch = torch_gqa(Q_torch.clone(), K_torch.clone(), V_torch.clone(), num_kv_heads, h)
            torch.cuda.synchronize()
            torch_time = (time.time() - start_time) * 1000 / test_runs
            
            speedup = torch_time / triton_time if triton_time > 0 else float('inf')
            
            config_str = f"B={B},N={N},d={d_model},h={h},kv={num_kv_heads}"
            correct_str = "✓" if is_correct else "✗"
            
            print(f"{config_str:<30} {triton_time:<12.3f} {torch_time:<14.3f} {speedup:<10.2f}x {max_diff:<12.2e} {correct_str:<8}")
            
        except Exception as e:
            config_str = f"B={B},N={N},d={d_model},h={h},kv={num_kv_heads}"
            print(f"{config_str:<30} {'ERROR':<12} {'ERROR':<14} {'ERROR':<10} {'ERROR':<12} {'✗':<8}")
            print(f"  Error: {str(e)}")
    
    print("-" * 90)
    
    # Memory usage analysis
    print("\nMemory Analysis (for largest successful config):")
    try:
        B, N, d_model, h, num_kv_heads = 1, 2048, 1024, 16, 8
        d_head = d_model // h
        kv_d_model = num_kv_heads * d_head
        
        Q = torch.randn(B, N, d_model, device='cuda', dtype=torch.float32)
        K = torch.randn(B, N, kv_d_model, device='cuda', dtype=torch.float32)
        V = torch.randn(B, N, kv_d_model, device='cuda', dtype=torch.float32)
        
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        
        # Triton memory
        start_mem = torch.cuda.memory_allocated()
        output_triton = torch.zeros_like(Q)
        solve(Q.clone(), K.clone(), V.clone(), output_triton, N, d_model, h, num_kv_heads)
        torch.cuda.synchronize()
        triton_mem = torch.cuda.max_memory_allocated() - start_mem
        
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        
        # PyTorch memory  
        start_mem = torch.cuda.memory_allocated()
        output_torch = torch_gqa(Q.clone(), K.clone(), V.clone(), num_kv_heads, h)
        torch.cuda.synchronize()
        torch_mem = torch.cuda.max_memory_allocated() - start_mem
        
        print(f"Triton peak memory: {triton_mem / 1024**2:.2f} MB")
        print(f"PyTorch peak memory: {torch_mem / 1024**2:.2f} MB")
        print(f"Memory ratio (PyTorch/Triton): {torch_mem / triton_mem:.2f}x")
        
    except Exception as e:
        print(f"Memory analysis failed: {e}")

if __name__ == "__main__":
    if torch.cuda.is_available():
        benchmark_gqa()
    else:
        print("CUDA not available, skipping benchmark")