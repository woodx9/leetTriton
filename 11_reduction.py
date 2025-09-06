import torch
import triton
import triton.language as tl

@triton.jit
def reduce_sum_kernel(a_ptr, b_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid_m = tl.program_id(axis=0)

    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs_m < N

    a = tl.load(a_ptr + offs_m, mask=mask, other=0)
    
    acc = tl.sum(a, axis=0) 

    # Store the partial sum (one per block)
    tl.store(b_ptr + pid_m, acc)


# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    BLOCK_SIZE = 64

    grid = (triton.cdiv(N, BLOCK_SIZE), )
    
    # Create temporary buffer for partial sums
    temp_input = input.detach().clone()
    temp_output = torch.zeros(grid[0], dtype=torch.float32, device='cuda')
    
    n = N
    print("n", n)
    while True:
        grid = (triton.cdiv(n, BLOCK_SIZE), )
        reduce_sum_kernel[grid](
            temp_input, temp_output,
            n,
            BLOCK_SIZE
        )
        print("temp_output", temp_output)
        
        temp_input = temp_output[:grid[0]].clone()

        temp_output.zero_()

        n = grid[0]
        if n <= 1:
            break

    output.copy_(temp_input[:1]) 




def test_reduction():
    """Test cases for the parallel reduction function"""
    
    print("Starting reduction tests...")
    
    # Test Case 1: Example from problem description
    print("\nTest Case 1: Basic example [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]")
    input_data = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=torch.float32, device='cuda')
    output = torch.zeros(1, dtype=torch.float32, device='cuda')
    solve(input_data, output, len(input_data))
    expected = 36.0
    print(f"Expected: {expected}, Got: {output[0].item()}")
    assert abs(output[0].item() - expected) < 1e-5, f"Test 1 failed: expected {expected}, got {output[0].item()}"
    
    # Test Case 2: Example with negative numbers
    print("\nTest Case 2: Mixed positive/negative [-2.5, 1.5, -1.0, 2.0]")
    input_data = torch.tensor([-2.5, 1.5, -1.0, 2.0], dtype=torch.float32, device='cuda')
    output = torch.zeros(1, dtype=torch.float32, device='cuda')
    solve(input_data, output, len(input_data))
    expected = 0.0
    print(f"Expected: {expected}, Got: {output[0].item()}")
    assert abs(output[0].item() - expected) < 1e-5, f"Test 2 failed: expected {expected}, got {output[0].item()}"
    
    # Test Case 3: Single element
    print("\nTest Case 3: Single element [42.5]")
    input_data = torch.tensor([42.5], dtype=torch.float32, device='cuda')
    output = torch.zeros(1, dtype=torch.float32, device='cuda')
    solve(input_data, output, len(input_data))
    expected = 42.5
    print(f"Expected: {expected}, Got: {output[0].item()}")
    assert abs(output[0].item() - expected) < 1e-5, f"Test 3 failed: expected {expected}, got {output[0].item()}"
    
    # Test Case 4: All zeros
    print("\nTest Case 4: All zeros [0.0, 0.0, 0.0, 0.0, 0.0]")
    input_data = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32, device='cuda')
    output = torch.zeros(1, dtype=torch.float32, device='cuda')
    solve(input_data, output, len(input_data))
    expected = 0.0
    print(f"Expected: {expected}, Got: {output[0].item()}")
    assert abs(output[0].item() - expected) < 1e-5, f"Test 4 failed: expected {expected}, got {output[0].item()}"
    
    # Test Case 5: All negative numbers
    print("\nTest Case 5: All negative [-1.0, -2.0, -3.0, -4.0]")
    input_data = torch.tensor([-1.0, -2.0, -3.0, -4.0], dtype=torch.float32, device='cuda')
    output = torch.zeros(1, dtype=torch.float32, device='cuda')
    solve(input_data, output, len(input_data))
    expected = -10.0
    print(f"Expected: {expected}, Got: {output[0].item()}")
    assert abs(output[0].item() - expected) < 1e-5, f"Test 5 failed: expected {expected}, got {output[0].item()}"
    
    print("\n✅ All tests passed successfully!")


if __name__ == "__main__":
    test_reduction()