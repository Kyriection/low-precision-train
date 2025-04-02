from torch.utils.cpp_extension import load
import torch

def benchmark_fp8(iterations = 100):
    fp8_ext = load(name="fp8_ext", sources=["./fp8_ext_kernel_eff.cu"], verbose=False)
    class FP8MatMulFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, A, B):
            # A and B are FP32.
            ctx.save_for_backward(A, B)
            # Call the FP4 matmul kernel (expects 2D Byte tensors).
            out = fp8_ext.fp8_matmul(A, A)
            return out
        
    print("Benchmarking FP8 matmul")    
    # Define larger matrix dimensions.
    M, K, N = 1024, 1024, 1024
    A = torch.randn(M, K, device="cuda", dtype=torch.float32)
    B = torch.randn(K, N, device="cuda", dtype=torch.float32)
    
    # Warm-up iterations.
    for _ in range(10):
        C = FP8MatMulFunction.apply(A,B)
        torch.cuda.synchronize()
    
    # Benchmark over multiple iterations using CUDA events.
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(iterations):
        C = FP8MatMulFunction.apply(A,B)
    end_event.record()
    torch.cuda.synchronize()
    
    elapsed_ms = start_event.elapsed_time(end_event)
    print("FP8 matmul average time: {:.3f} ms".format(elapsed_ms / iterations))
    
def benchmark_fp4_eff(iterations = 100):
    fp4_ext = load(name="fp4_ext", sources=["./fp4_ext_kernel_eff.cu"], verbose=False)
    class FP4MatMulFunctionEFF(torch.autograd.Function):
        @staticmethod
        def forward(ctx, A, B):
            # A and B are FP32.
            ctx.save_for_backward(A, B)
            # Call the FP4 matmul kernel (expects 2D Byte tensors).
            out = fp4_ext.fp4_matmul(A, A)
            return out
        
    print("Benchmarking FP4 matmul")
    M, K, N = 1024, 1024, 1024
    A = torch.randn(M, K, device="cuda", dtype=torch.float32)
    B = torch.randn(K, N, device="cuda", dtype=torch.float32)
    
    # Warm-up iterations.
    for _ in range(10):
        C = FP4MatMulFunctionEFF.apply(A, B)
        torch.cuda.synchronize()
    
    # Benchmark over multiple iterations using CUDA events.
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(iterations):
        C = FP4MatMulFunctionEFF.apply(A, B)
    end_event.record()
    torch.cuda.synchronize()
    
    elapsed_ms = start_event.elapsed_time(end_event)
    print("FP4 matmul average time: {:.3f} ms".format(elapsed_ms / iterations))

def benchmark_fp4(iterations = 100):
    from fp4_torch_kernel.utils import FP4MatMulFunction
    print("Benchmarking FP4 matmul")
    M, K, N = 1024, 1024, 1024
    A = torch.randn(M, K, device="cuda", dtype=torch.float32)
    B = torch.randn(K, N, device="cuda", dtype=torch.float32)
    
    # Warm-up iterations.
    for _ in range(10):
        C = FP4MatMulFunction.apply(A, B)
        torch.cuda.synchronize()
    
    # Benchmark over multiple iterations using CUDA events.
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(iterations):
        C = FP4MatMulFunction.apply(A, B)
    end_event.record()
    torch.cuda.synchronize()
    
    elapsed_ms = start_event.elapsed_time(end_event)
    print("FP4 matmul average time: {:.3f} ms".format(elapsed_ms / iterations))

def benchmark_fp8_native(iterations = 100):
    print("Benchmarking FP16 matmul")
    M, K, N = 1024, 1024, 1024
    # Create matrices directly in half precision.
    A = torch.randn(M, K, device="cuda", dtype=torch.floaat32).to(torch.float8_e5m2)
    B = torch.randn(K, N, device="cuda", dtype=torch.float32).to(torch.float8_e5m2)

    # Warm-up iterations.
    for _ in range(10):
        torch.ao()
        C = torch.add(A*B.T)
        C = torch.matmul(A, B)
        torch.cuda.synchronize()

    # Benchmark over multiple iterations using CUDA events.
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(iterations):
        C = torch.matmul(A, B)
    end_event.record()
    torch.cuda.synchronize()

    elapsed_ms = start_event.elapsed_time(end_event)
    print("FP16 matmul average time: {:.3f} ms".format(elapsed_ms / iterations))

def benchmark_fp16(iterations = 100):
    print("Benchmarking FP16 matmul")
    M, K, N = 1024, 1024, 1024
    # Create matrices directly in half precision.
    A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)

    # Warm-up iterations.
    for _ in range(10):
        C = torch.matmul(A, B)
        torch.cuda.synchronize()

    # Benchmark over multiple iterations using CUDA events.
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(iterations):
        C = torch.matmul(A, B)
    end_event.record()
    torch.cuda.synchronize()

    elapsed_ms = start_event.elapsed_time(end_event)
    print("FP16 matmul average time: {:.3f} ms".format(elapsed_ms / iterations))

def benchmark_fp32(iterations = 100):
    print("Benchmarking FP32 matmul")
    M, K, N = 1024, 1024, 1024
    # Create matrices directly in half precision.
    A = torch.randn(M, K, device="cuda", dtype=torch.float32)
    B = torch.randn(K, N, device="cuda", dtype=torch.float32)

    # Warm-up iterations.
    for _ in range(10):
        C = torch.matmul(A, B)
        torch.cuda.synchronize()

    # Benchmark over multiple iterations using CUDA events.
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(iterations):
        C = torch.matmul(A, B)
    end_event.record()
    torch.cuda.synchronize()

    elapsed_ms = start_event.elapsed_time(end_event)
    print("FP32 matmul average time: {:.3f} ms".format(elapsed_ms / iterations))

if __name__ == "__main__":
    # print("Bench 32")
    # benchmark_fp32(1000)
    # print("Bench 16")
    # benchmark_fp16(1000)
    # print("Bench 8 native")
    # # benchmark_fp8_native(1000)
    # print("Bench 8")
    # benchmark_fp8(1000)
    # print("Bench 4 eff")
    # benchmark_fp4_eff(1000)
    print("Bench 4")
    benchmark_fp4(10000)

"""
Analyzed timings A100:
FP32 matmul average time: 0.129 ms
FP16 matmul average time: 0.019 ms
FP8 matmul average time: 0.019 ms
FP4 matmul average time: 0.018 ms

Analyzed timings H100:
FP32 matmul average time: 0.056 ms
FP16 matmul average time: 0.011 ms
FP8 matmul average time: 0.010 ms
FP4 matmul average time: 0.439 ms

"""

