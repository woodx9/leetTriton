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
def solve(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, output: torch.Tensor, N: int, d_model: int, h: int):
    """
    Q, K, V: [B, N, d_model]
    output:  [B, N, d_model]
    N: sequence length
    d_model: model dim
    h: num heads
    """
     
    B = Q.shape[0]
    d_head = d_model // h
    M = N   # query_len == key_len

    # reshape to multi-head form [B, h, N, d_head]
    Qh = Q.view(B, N, h, d_head).transpose(1, 2).contiguous()  # [B, h, N, d_head]
    Kh = K.view(B, N, h, d_head).transpose(1, 2).contiguous()  # [B, h, N, d_head]
    Vh = V.view(B, N, h, d_head).transpose(1, 2).contiguous()  # [B, h, N, d_head]
    Oh = output.view(B, N, h, d_head).transpose(1, 2).contiguous()  # [B, h, N, d_head]

    # flatten batch*head
    Qf = Qh.reshape(B * h, N, d_head)
    Kf = Kh.reshape(B * h, N, d_head)
    Vf = Vh.reshape(B * h, N, d_head)
    Of = Oh.reshape(B * h, N, d_head)

    grid = (B * h * M,)
    softmax_attention_kernel[grid](
        Qf, Kf, Vf, Of,
        M, N, d_head,
        BLOCK_SIZE_M=1,
        BLOCK_SIZE_N=triton.next_power_of_2(min(128, N)),
        BLOCK_SIZE_D=triton.next_power_of_2(d_head),
    )

    # reshape back to [B, N, d_model]
    output.copy_(Oh.transpose(1, 2).reshape(B, N, d_model))


def benchmark_multi_head_attention():
    """
    测试Triton实现的Multi Head Attention与PyTorch官方MultiheadAttention的性能对比
    覆盖不同的矩阵大小范围
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("CUDA is not available, running on CPU")
        return
    
    print("Multi Head Attention Performance Benchmark")
    print("Testing with real-world model configurations:")
    print("- GPT-2 Small/Medium/Large")
    print("- BERT Base/Large") 
    print("- LLaMA configurations")
    print("=" * 80)
    print(f"{'Batch Size':<10} {'Seq Len':<8} {'d_model':<8} {'Heads':<6} {'Model Type':<15} {'Triton (ms)':<12} {'PyTorch (ms)':<13} {'Speedup':<8}")
    print("-" * 80)
    
    # 测试配置: (batch_size, seq_len, d_model, num_heads, model_type)
    # 基于流行模型的真实配置
    test_configs = [
        # GPT-2 Small配置 (d_model=768, heads=12)
        (1, 512, 768, 12, "GPT-2 Small"),
        (2, 512, 768, 12, "GPT-2 Small"),
        (4, 512, 768, 12, "GPT-2 Small"),
        (1, 1024, 768, 12, "GPT-2 Small"),
        
        # GPT-2 Medium配置 (d_model=1024, heads=16)
        (1, 512, 1024, 16, "GPT-2 Medium"),
        (2, 512, 1024, 16, "GPT-2 Medium"),
        (1, 1024, 1024, 16, "GPT-2 Medium"),
        
        # GPT-2 Large配置 (d_model=1280, heads=20)
        (1, 512, 1280, 20, "GPT-2 Large"),
        (2, 512, 1280, 20, "GPT-2 Large"),
        
        # BERT Base配置 (d_model=768, heads=12)
        (1, 256, 768, 12, "BERT Base"),
        (8, 256, 768, 12, "BERT Base"),
        (16, 256, 768, 12, "BERT Base"),
        
        # BERT Large配置 (d_model=1024, heads=16)
        (1, 256, 1024, 16, "BERT Large"),
        (8, 256, 1024, 16, "BERT Large"),
        
        # LLaMA配置 (d_model=4096, heads=32)
        (1, 512, 4096, 32, "LLaMA"),
        (2, 512, 4096, 32, "LLaMA"),
        
        # 较小的测试配置
        (1, 128, 512, 8, "Small Test"),
        (1, 256, 512, 8, "Small Test"),
        (4, 128, 512, 8, "Small Test"),
    ]
    
    warmup_runs = 5
    test_runs = 20
    
    for B, N, d_model, h, model_type in test_configs:
        try:
            # 确保d_model能被h整除
            if d_model % h != 0:
                continue
                
            d_head = d_model // h
            
            # 创建测试数据
            Q = torch.randn(B, N, d_model, device=device, dtype=torch.float32)
            K = torch.randn(B, N, d_model, device=device, dtype=torch.float32)
            V = torch.randn(B, N, d_model, device=device, dtype=torch.float32)
            
            # Triton版本输出
            output_triton = torch.zeros_like(Q)
            
            # PyTorch版本
            multihead_attn = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=h,
                batch_first=True,
                bias=False
            ).to(device)
            
            # 设置权重为单位矩阵 (简化比较)
            with torch.no_grad():
                multihead_attn.in_proj_weight.copy_(torch.eye(3 * d_model, d_model).to(device))
                multihead_attn.out_proj.weight.copy_(torch.eye(d_model).to(device))
            
            # Warmup
            for _ in range(warmup_runs):
                # Triton warmup
                output_triton.zero_()
                solve(Q, K, V, output_triton, N, d_model, h)
                
                # PyTorch warmup
                with torch.no_grad():
                    _ = multihead_attn(Q, K, V, need_weights=False)[0]
            
            torch.cuda.synchronize()
            
            # 测试Triton性能
            start_time = time.time()
            for _ in range(test_runs):
                output_triton.zero_()
                solve(Q, K, V, output_triton, N, d_model, h)
            torch.cuda.synchronize()
            triton_time = (time.time() - start_time) * 1000 / test_runs
            
            # 测试PyTorch性能
            start_time = time.time()
            for _ in range(test_runs):
                with torch.no_grad():
                    output_torch = multihead_attn(Q, K, V, need_weights=False)[0]
            torch.cuda.synchronize()
            torch_time = (time.time() - start_time) * 1000 / test_runs
            
            # 计算加速比
            speedup = torch_time / triton_time if triton_time > 0 else 0
            
            print(f"{B:<10} {N:<8} {d_model:<8} {h:<6} {model_type:<15} {triton_time:<12.2f} {torch_time:<13.2f} {speedup:<8.2f}x")
            
        except Exception as e:
            print(f"Error with config (B={B}, N={N}, d_model={d_model}, h={h}, {model_type}): {str(e)}")
            continue
    
    print("-" * 80)
    print("Note: Speedup = PyTorch time / Triton time")
    print("Values > 1.0 indicate Triton is faster")


if __name__ == "__main__":
    benchmark_multi_head_attention()