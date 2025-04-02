#include <cuda_fp8.h>
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_SIZE 256

// Helper to load a tile of matrix A using cp.async with a 16-byte (float4) transfer.
// Each tile is TILE_SIZE x TILE_SIZE floats, which is 1024 floats or 256 float4 elements.
// Only threads with linear index < 256 perform a load.
__device__ inline void load_A_tile_async(const float* A, float* tile, int M, int K, int block_row, int tile_idx) {
  int tid = threadIdx.y * TILE_SIZE + threadIdx.x;
  if (tid < 256) {
    // Each row of the tile is 32 floats, which is 8 groups of 4 floats.
    int row_in_tile = tid / (TILE_SIZE / 4);  // (TILE_SIZE/4)==8
    int col_group    = tid % (TILE_SIZE / 4);
    int global_row   = block_row * TILE_SIZE + row_in_tile;
    int global_col   = tile_idx * TILE_SIZE + col_group * 4;
    float4* dest = reinterpret_cast<float4*>(tile);
    // Only load if within bounds.
    if (global_row < M && (global_col + 3) < K) {
      const float4* src = reinterpret_cast<const float4*>(A + global_row * K + global_col);
      unsigned long long dest_ll = reinterpret_cast<unsigned long long>(&dest[tid]);
      unsigned long long src_ll  = reinterpret_cast<unsigned long long>(src);
      asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n"
                   :
                   : "l"(dest_ll), "l"(src_ll), "n"(16)
                   : "memory");
    } else {
      float4 zero = {0.f, 0.f, 0.f, 0.f};
      reinterpret_cast<float4*>(tile)[tid] = zero;
    }
  }
}

// Helper to load a tile of matrix B using cp.async with a 16-byte (float4) transfer.
__device__ inline void load_B_tile_async(const float* B, float* tile, int K, int N, int block_col, int tile_idx) {
  int tid = threadIdx.y * TILE_SIZE + threadIdx.x;
  if (tid < 256) {
    int row_in_tile = tid / (TILE_SIZE / 4);
    int col_group   = tid % (TILE_SIZE / 4);
    int global_row  = tile_idx * TILE_SIZE + row_in_tile;
    int global_col  = block_col * TILE_SIZE + col_group * 4;
    float4* dest = reinterpret_cast<float4*>(tile);
    if (global_row < K && (global_col + 3) < N) {
      const float4* src = reinterpret_cast<const float4*>(B + global_row * N + global_col);
      unsigned long long dest_ll = reinterpret_cast<unsigned long long>(&dest[tid]);
      unsigned long long src_ll  = reinterpret_cast<unsigned long long>(src);
      asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n"
                   :
                   : "l"(dest_ll), "l"(src_ll), "n"(16)
                   : "memory");
    } else {
      float4 zero = {0.f, 0.f, 0.f, 0.f};
      reinterpret_cast<float4*>(tile)[tid] = zero;
    }
  }
}

__global__ void fp8_matmul_kernel(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C,
                                  int M, int K, int N) {
  // Compute output element indices.
  int row = blockIdx.y * TILE_SIZE + threadIdx.y;
  int col = blockIdx.x * TILE_SIZE + threadIdx.x;
  float acc = 0.0f;

  // Allocate shared memory for two double-buffered tiles for A and two for B.
  extern __shared__ float shared_mem[];
  float* A_tile0 = shared_mem;
  float* B_tile0 = A_tile0 + TILE_SIZE * TILE_SIZE;
  float* A_tile1 = B_tile0 + TILE_SIZE * TILE_SIZE;
  float* B_tile1 = A_tile1 + TILE_SIZE * TILE_SIZE;

  // Pointers for current and next buffers.
  float* curr_A_tile = A_tile0;
  float* curr_B_tile = B_tile0;
  float* next_A_tile = A_tile1;
  float* next_B_tile = B_tile1;

  int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

  // Preload first tile (t = 0) for A and B.
  if (threadIdx.y * TILE_SIZE + threadIdx.x < 256) {
    load_A_tile_async(A, curr_A_tile, M, K, blockIdx.y, 0);
    load_B_tile_async(B, curr_B_tile, K, N, blockIdx.x, 0);
  }
  asm volatile("cp.async.wait_all;" ::: "memory");
  __syncthreads();

  for (int t = 0; t < numTiles; t++) {
    int next_t = t + 1;
    if (next_t < numTiles) {
      // Asynchronously prefetch the next tiles into the "next" buffers.
      if (threadIdx.y * TILE_SIZE + threadIdx.x < 256) {
        load_A_tile_async(A, next_A_tile, M, K, blockIdx.y, next_t);
        load_B_tile_async(B, next_B_tile, K, N, blockIdx.x, next_t);
      }
    }
    asm volatile("cp.async.wait_all;" ::: "memory");
    __syncthreads();

    // Compute using the current tile.
    for (int k = 0; k < TILE_SIZE; ++k) {
      float a_val = curr_A_tile[threadIdx.y * TILE_SIZE + k];
      float b_val = curr_B_tile[k * TILE_SIZE + threadIdx.x];
      // Optionally perform fused FP8 conversion.
      __nv_fp8_e4m3 a_fp8 = __nv_fp8_e4m3(a_val);
      __nv_fp8_e4m3 b_fp8 = __nv_fp8_e4m3(b_val);
      float a_conv = static_cast<float>(a_fp8);
      float b_conv = static_cast<float>(b_fp8);
      acc += a_conv * b_conv;
    }
    __syncthreads();

    // Swap current and next buffers.
    float* tempA = curr_A_tile;
    float* tempB = curr_B_tile;
    curr_A_tile = next_A_tile;
    curr_B_tile = next_B_tile;
    next_A_tile = tempA;
    next_B_tile = tempB;
  }

  if (row < M && col < N) {
    __nv_fp8_e4m3 c_fp8 = __nv_fp8_e4m3(acc);
    C[row * N + col] = static_cast<float>(c_fp8);
  }
}

torch::Tensor fp8_matmul(torch::Tensor A, torch::Tensor B) {
  int M = A.size(0);
  int K = A.size(1);
  int N = B.size(1);

  auto C = torch::empty({M, N}, torch::dtype(torch::kFloat32).device(A.device()));

  dim3 threads(TILE_SIZE, TILE_SIZE);
  dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

  // Allocate shared memory for 4 tiles (each TILE_SIZE×TILE_SIZE floats).
  size_t shared_mem_size = 4 * TILE_SIZE * TILE_SIZE * sizeof(float);

  fp8_matmul_kernel<<<blocks, threads, shared_mem_size>>>(
      A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, K, N);
  return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fp8_matmul", &fp8_matmul, "Optimized fused FP8 matmul with vectorized cp.async, double buffering, and unrolling");
}
