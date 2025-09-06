import torch
import triton
import triton.language as tl

@triton.jit
def sigmoid_linear_unit_kernel(a_ptr, b_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid_m = tl.program_id(axis=0)

    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs_m < N

    a = tl.load(a_ptr + offs_m, mask=mask, other=0)
    
    sigmoid = 1 / (1 + tl.exp(-a))
    b = a * sigmoid

    tl.store(b_ptr + offs_m, b, mask=mask)


# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    BLOCK_SIZE = 16 
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    sigmoid_linear_unit_kernel[grid](
        input, output,
        N,
        BLOCK_SIZE
    )


def test_sigmoid_linear_unit():
    """测试 sigmoid linear unit 函数的多个测试用例"""
    print("开始测试 Sigmoid Linear Unit...")
    
    # 测试用例1: 基本功能测试
    print("\n测试用例1: 基本功能测试")
    N = 8
    input_data = torch.tensor([0.0, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 3.0], dtype=torch.float32, device='cuda')
    output = torch.zeros(N, dtype=torch.float32, device='cuda')
    
    solve(input_data, output, N)
    
    # 计算期望结果 (sigmoid linear unit: x * sigmoid(x))
    expected = input_data * torch.sigmoid(input_data)
    
    print(f"输入: {input_data.cpu()}")
    print(f"输出: {output.cpu()}")
    print(f"期望: {expected.cpu()}")
    print(f"误差: {torch.abs(output - expected).cpu()}")
    print(f"最大误差: {torch.max(torch.abs(output - expected)).item():.6f}")
    
    # 测试用例2: 较大的数组
    print("\n测试用例2: 较大数组测试 (N=100)")
    N = 100
    input_data = torch.randn(N, dtype=torch.float32, device='cuda')
    output = torch.zeros(N, dtype=torch.float32, device='cuda')
    
    solve(input_data, output, N)
    expected = input_data * torch.sigmoid(input_data)
    
    max_error = torch.max(torch.abs(output - expected)).item()
    print(f"最大误差: {max_error:.6f}")
    print(f"平均误差: {torch.mean(torch.abs(output - expected)).item():.6f}")
    
    # 测试用例3: 边界值测试
    print("\n测试用例3: 边界值测试")
    N = 6
    input_data = torch.tensor([0.0, 10.0, -10.0, 1e-6, -1e-6, 0.0], dtype=torch.float32, device='cuda')
    output = torch.zeros(N, dtype=torch.float32, device='cuda')
    
    solve(input_data, output, N)
    expected = input_data * torch.sigmoid(input_data)
    
    print(f"输入: {input_data.cpu()}")
    print(f"输出: {output.cpu()}")
    print(f"期望: {expected.cpu()}")
    print(f"最大误差: {torch.max(torch.abs(output - expected)).item():.6f}")
    
    # 测试用例4: 非16的倍数长度测试
    print("\n测试用例4: 非16倍数长度测试 (N=25)")
    N = 25
    input_data = torch.linspace(-3, 3, N, dtype=torch.float32, device='cuda')
    output = torch.zeros(N, dtype=torch.float32, device='cuda')
    
    solve(input_data, output, N)
    expected = input_data * torch.sigmoid(input_data)
    
    max_error = torch.max(torch.abs(output - expected)).item()
    print(f"最大误差: {max_error:.6f}")
    print(f"平均误差: {torch.mean(torch.abs(output - expected)).item():.6f}")
    
    # 性能测试
    print("\n性能测试 (N=10000)")
    N = 10000
    input_data = torch.randn(N, dtype=torch.float32, device='cuda')
    output = torch.zeros(N, dtype=torch.float32, device='cuda')
    
    # 预热
    for _ in range(10):
        solve(input_data, output, N)
    torch.cuda.synchronize()
    
    # 计时
    import time
    start_time = time.time()
    for _ in range(100):
        solve(input_data, output, N)
    torch.cuda.synchronize()
    end_time = time.time()
    
    avg_time = (end_time - start_time) / 100 * 1000  # 转换为毫秒
    print(f"平均执行时间: {avg_time:.3f} ms")
    
    print("\n所有测试完成!")


if __name__ == "__main__":
    # 检查CUDA是否可用
    if not torch.cuda.is_available():
        print("错误: CUDA不可用，无法运行测试")
        exit(1)
    
    test_sigmoid_linear_unit()