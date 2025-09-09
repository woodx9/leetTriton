import torch
import triton
import triton.language as tl
import time
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

# 简单处理，复用softmax attention
@triton.jit
def softmax_attention_kernel(
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    B_H, M, N, D,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr
):
    pid = tl.program_id(0)
    
    # 添加这两行来正确处理索引
    batch_head_idx = pid // M  
    row = pid % M              
    
    # 如果超出范围就返回
    if batch_head_idx >= B_H:
        return

    offs_d = tl.arange(0, BLOCK_SIZE_D)
    mask_q = offs_d < D
    
    # 修改数据访问：加上batch_head偏移
    q_offset = batch_head_idx * M * D + row * D
    q = tl.load(Q_ptr + q_offset + offs_d, mask=mask_q, other=0.0)
    q = tl.reshape(q, [1, BLOCK_SIZE_D])

    out = tl.zeros([BLOCK_SIZE_D], dtype=tl.float32)
    
    global_max = -float("inf")
    total_exp_sum = 0.0
    scale = tl.sqrt(tl.cast(D, tl.float32))

    for j in range(0, N, BLOCK_SIZE_N):
        offs_n = j + tl.arange(0, BLOCK_SIZE_N)
        mask_kv = (offs_n[:, None] < N) & (offs_d[None, :] < D)
        
        k_base_offset = batch_head_idx * N * D
        k = tl.load(K_ptr + k_base_offset + offs_n[:, None] * D + offs_d[None, :], mask=mask_kv, other=0.0)
        s = tl.sum(k * q, axis=1) / scale
        
        mask_n = offs_n < N
        s = tl.where(mask_n, s, -float("inf"))
        
        local_max = tl.max(s, axis=0)
        global_max = tl.maximum(global_max, local_max)

    for j in range(0, N, BLOCK_SIZE_N):
        offs_n = j + tl.arange(0, BLOCK_SIZE_N)
        mask_kv = (offs_n[:, None] < N) & (offs_d[None, :] < D)
        
        k_base_offset = batch_head_idx * N * D
        v_base_offset = batch_head_idx * N * D
        k = tl.load(K_ptr + k_base_offset + offs_n[:, None] * D + offs_d[None, :], mask=mask_kv, other=0.0)
        v = tl.load(V_ptr + v_base_offset + offs_n[:, None] * D + offs_d[None, :], mask=mask_kv, other=0.0)
        
        s = tl.sum(k * q, axis=1) / scale
        mask_n = offs_n < N
        s = tl.where(mask_n, s, -float("inf"))
        
        exp_s = tl.exp(s - global_max)
        total_exp_sum += tl.sum(exp_s, axis=0)
        
        w = tl.reshape(exp_s, [BLOCK_SIZE_N, 1])
        out += tl.sum(w * v, axis=0)

    out = out / total_exp_sum
    
    o_offset = batch_head_idx * M * D + row * D
    tl.store(Out_ptr + o_offset + offs_d, out, mask=mask_q)

# Q, K, V, output are tensors on the GPU
# Q, K, V [B, h, N, d_head] 
# d_model // h = d_head
def solve(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, output: torch.Tensor, N: int, d_model: int, h: int):
    B = Q.shape[0]
    d_head = d_model // h
    M = N

    # [B, h, N, d_head]
    Qh = Q.contiguous()
    Kh = K.contiguous()
    Vh = V.contiguous()
    Oh = output.contiguous()

    # flatten batch*head
    Qf = Qh.reshape(B * h, N, d_head)
    Kf = Kh.reshape(B * h, N, d_head)
    Vf = Vh.reshape(B * h, N, d_head)
    Of = Oh.reshape(B * h, N, d_head)

    grid = (B * h * M,)
    softmax_attention_kernel[grid](
        Qf, Kf, Vf, Of,
        B * h, M, N, d_head,
        BLOCK_SIZE_M=1,
        BLOCK_SIZE_N=triton.next_power_of_2(min(128, N)),
        BLOCK_SIZE_D=triton.next_power_of_2(d_head),
    )
    # reshape back
    output.copy_(Oh)

class TritonMHA(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model) 
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def forward(self, query, key, value):
        batch_size, seq_len = query.size(0), query.size(1)
        
        # 1. 输入投影
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)
        
        # 2. 重塑为多头形式
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # 确保连续性
        Q = Q.contiguous()
        K = K.contiguous()
        V = V.contiguous()
        
        # 3. Triton注意力计算
        # 3. Triton注意力计算
        attention_output = torch.zeros_like(Q)
        solve(Q, K, V, attention_output, seq_len, self.d_model, self.num_heads)
        
        # 4. 合并多头
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, seq_len, self.d_model)
        
        # 5. 输出投影
        output = self.W_o(attention_output)
        
        return output, None

def benchmark_torch_vs_triton_mha():
    """
    比较PyTorch标准MultiHeadAttention和Triton TritonMHA的性能和精度
    使用与debug函数相同的调用方式确保公平对比
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("CUDA is not available, running on CPU")
        return
    
    print("PyTorch vs Triton MultiHeadAttention Benchmark")
    print("=" * 100)
    print(f"{'Batch Size':<10} {'Seq Len':<8} {'d_model':<8} {'Heads':<6} {'Model Type':<15} {'PyTorch (ms)':<13} {'Triton (ms)':<12} {'Speedup':<8} {'Max Error':<12} {'Mean Error':<12}")
    print("-" * 100)
    
    # 测试配置
    test_configs = [
        # 小规模测试
        (1, 32, 256, 8, "Small"),
        (2, 64, 256, 8, "Small"),
        
        # 中规模测试
        (1, 128, 512, 8, "Medium"),
        (2, 256, 512, 8, "Medium"),
        
        # GPT-2 Small风格
        (1, 256, 768, 12, "GPT-2 Small"),
        (2, 512, 768, 12, "GPT-2 Small"),
        
        # BERT Base风格
        (4, 128, 768, 12, "BERT Base"),
        
        # 更大规模测试
        (1, 512, 1024, 16, "Large"),
    ]
    
    warmup_runs = 3
    test_runs = 10
    
    for B, N, d_model, h, model_type in test_configs:
        try:
            # 确保d_model能被h整除
            if d_model % h != 0:
                continue
                
            # 创建测试数据
            torch.manual_seed(42)  # 固定随机种子确保可重复性
            x = torch.randn(B, N, d_model, device=device, dtype=torch.float32)
            
            # 创建模型 - 使用与debug相同的方式
            torch_mha = nn.MultiheadAttention(
                embed_dim=d_model, 
                num_heads=h, 
                dropout=0.0,
                batch_first=True
            ).to(device)
            
            triton_mha = TritonMHA(d_model, h, dropout=0.0).to(device)
            
            # 权重同步 - 按照debug函数的方式
            with torch.no_grad():
                # PyTorch MHA使用的是in_proj_weight和out_proj
                # 需要将其分解为Q、K、V权重
                if hasattr(torch_mha, 'in_proj_weight') and torch_mha.in_proj_weight is not None:
                    # 分解in_proj_weight
                    W_q_weight = torch_mha.in_proj_weight[:d_model, :]
                    W_k_weight = torch_mha.in_proj_weight[d_model:2*d_model, :]  
                    W_v_weight = torch_mha.in_proj_weight[2*d_model:, :]
                    
                    triton_mha.W_q.weight.copy_(W_q_weight)
                    triton_mha.W_k.weight.copy_(W_k_weight)
                    triton_mha.W_v.weight.copy_(W_v_weight)
                    
                    # bias处理
                    if torch_mha.in_proj_bias is not None:
                        triton_mha.W_q.bias.copy_(torch_mha.in_proj_bias[:d_model])
                        triton_mha.W_k.bias.copy_(torch_mha.in_proj_bias[d_model:2*d_model])
                        triton_mha.W_v.bias.copy_(torch_mha.in_proj_bias[2*d_model:])
                    
                    # 输出投影
                    triton_mha.W_o.weight.copy_(torch_mha.out_proj.weight)
                    if torch_mha.out_proj.bias is not None:
                        triton_mha.W_o.bias.copy_(torch_mha.out_proj.bias)
            
            # 设置为评估模式
            torch_mha.eval()
            triton_mha.eval()
            
            # Warmup
            for _ in range(warmup_runs):
                with torch.no_grad():
                    # PyTorch warmup
                    _ = torch_mha(x, x, x, need_weights=False)[0]
                    
                    # Triton warmup  
                    _ = triton_mha(x, x, x)[0]
            
            torch.cuda.synchronize()
            
            # 测试PyTorch性能
            start_time = time.time()
            for _ in range(test_runs):
                with torch.no_grad():
                    torch_output = torch_mha(x, x, x, need_weights=False)[0]
            torch.cuda.synchronize()
            torch_time = (time.time() - start_time) * 1000 / test_runs
            
            # 测试Triton性能
            start_time = time.time()
            for _ in range(test_runs):
                with torch.no_grad():
                    triton_output = triton_mha(x, x, x)[0]
            torch.cuda.synchronize()
            triton_time = (time.time() - start_time) * 1000 / test_runs
            
            # 计算数值误差 - 重新计算确保结果准确
            with torch.no_grad():
                torch_final = torch_mha(x, x, x, need_weights=False)[0]
                triton_final = triton_mha(x, x, x)[0]
            
            # 计算误差
            abs_error = torch.abs(torch_final - triton_final)
            max_error = torch.max(abs_error).item()
            mean_error = torch.mean(abs_error).item()
            
            # 计算加速比
            speedup = torch_time / triton_time if triton_time > 0 else 0
            
            print(f"{B:<10} {N:<8} {d_model:<8} {h:<6} {model_type:<15} {torch_time:<13.2f} {triton_time:<12.2f} {speedup:<8.2f}x {max_error:<12.2e} {mean_error:<12.2e}")
            
            # 如果误差过大，打印警告
            if max_error > 1e-4:
                print(f"  WARNING: Large numerical error detected! Max: {max_error:.2e}, Mean: {mean_error:.2e}")
            
        except Exception as e:
            print(f"Error with config (B={B}, N={N}, d_model={d_model}, h={h}, {model_type}): {str(e)}")
            continue
    
    print("-" * 100)
    print("Note:")
    print("- Speedup = PyTorch time / Triton time (values > 1.0 indicate Triton is faster)")
    print("- Max Error: Maximum absolute difference between outputs")
    print("- Mean Error: Average absolute difference between outputs") 
    print("- Weights are synchronized to ensure fair comparison")


if __name__ == "__main__":
    benchmark_torch_vs_triton_mha()
