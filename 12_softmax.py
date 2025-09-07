import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    a_ptr, b_ptr,
    N,
    BLOCK_SIZE: tl.constexpr
):
    row = tl.program_id(0)
    row_start = row * N

    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    x = tl.load(a_ptr + row_start + offs, mask=mask, other=-float("inf"))

    x_max = tl.max(x)
    x = x - x_max

    x_exp = tl.exp(x)
    denorm = tl.sum(x_exp)

    y = x_exp / denorm

    tl.store(b_ptr + row_start + offs, y, mask=mask)

# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    BLOCK_SIZE = triton.next_power_of_2(N)

    B, N = input.shape
    grid = (B, )
    softmax_kernel[grid](input, output, N, BLOCK_SIZE)

def benchmark():
    import time
    import numpy as np
    
    print("Benchmarking Triton vs PyTorch Softmax Performance")
    print("=" * 80)
    print(f"{'Matrix Size':<15} {'PyTorch (ms)':<15} {'Triton (ms)':<15} {'Speedup':<15} {'Max Diff'}")
    print("-" * 80)
    
    # Test different matrix sizes
    sizes = [
    (128, 256),        # Small
    (256, 512),        # Medium small
    (512, 1024),       # Medium
    (1024, 2048),      # Large
    (2048, 4096),      # Very large
    (4096, 8192),      # Extra large
    (8192, 8192),      # Square big
    (16384, 16384),    # Huge square
    (32768, 4096),     # Tall matrix
    (4096, 32768),     # Wide matrix
    (65536, 1024),     # Very tall
    (1024, 65536),     # Very wide
]
    warmup_runs = 5
    benchmark_runs = 100
    
    for B, N in sizes:
        # Create input tensor
        input_tensor = torch.randn(B, N, device='cuda', dtype=torch.float32)
        output_tensor = torch.empty_like(input_tensor)
        
        # Warmup runs
        for _ in range(warmup_runs):
            # PyTorch warmup
            torch.softmax(input_tensor, dim=1)
            
            # Triton warmup
            solve(input_tensor, output_tensor, N)
        
        torch.cuda.synchronize()
        
        # Benchmark PyTorch softmax
        start_time = time.perf_counter()
        for _ in range(benchmark_runs):
            torch_result = torch.softmax(input_tensor, dim=1)
        torch.cuda.synchronize()
        pytorch_time = (time.perf_counter() - start_time) * 1000 / benchmark_runs
        
        # Benchmark Triton softmax
        start_time = time.perf_counter()
        for _ in range(benchmark_runs):
            solve(input_tensor, output_tensor, N)
        torch.cuda.synchronize()
        triton_time = (time.perf_counter() - start_time) * 1000 / benchmark_runs
        
        # Calculate speedup
        speedup = pytorch_time / triton_time if triton_time > 0 else 0
        
        # Verify correctness
        solve(input_tensor, output_tensor, N)
        torch_result = torch.softmax(input_tensor, dim=1)
        max_diff = torch.max(torch.abs(output_tensor - torch_result)).item()
        
        print(f"{B}x{N:<10} {pytorch_time:<15.3f} {triton_time:<15.3f} {speedup:<10.2f}x (max_diff: {max_diff:.2e})")
        
        # Check if results are close enough
        if max_diff > 1e-4:
            print(f"Warning: Large difference detected: {max_diff}")
    
    print("-" * 80)
    print("Benchmark completed!")

if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    benchmark()
