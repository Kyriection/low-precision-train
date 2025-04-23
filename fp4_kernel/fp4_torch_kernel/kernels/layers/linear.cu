#include <cuda_fp4.h>
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_SIZE 64      // For simplicity; adjust if you implement per-thread multi-load for larger tiles.
#define UNROLL_FACTOR 4   // Adjust to balance loop overhead vs. register pressure.

// -------------------------
// Fused Linear Kernel (Forward)
// Computes: Y = X * W + bias, where X: (M x K), W: (K x N), bias: (N)
// FP4 arithmetic is simulated via conversion (using __nv_fp4_e2m1).
// -------------------------
__global__ void fp4_linear_kernel(const float* __restrict__ X,
                                    const float* __restrict__ W,
                                    const float* __restrict__ bias,
                                    float* __restrict__ Y,
                                    int M, int K, int N) {
  // Each thread computes one output element.
  const int row = blockIdx.y * TILE_SIZE + threadIdx.y;  // M index
  const int col = blockIdx.x * TILE_SIZE + threadIdx.x;  // N index

  float acc = 0.0f;

  // Allocate shared memory for double buffering.
  // We allocate 4 tiles: two for X and two for W.
  extern __shared__ float shared_mem[];
  float* X_tile0 = shared_mem;
  float* W_tile0 = X_tile0 + TILE_SIZE * TILE_SIZE;
  float* X_tile1 = W_tile0 + TILE_SIZE * TILE_SIZE;
  float* W_tile1 = X_tile1 + TILE_SIZE * TILE_SIZE;

  // Pointers to current and next buffers.
  float* curr_X_tile = X_tile0;
  float* curr_W_tile = W_tile0;
  float* next_X_tile = X_tile1;
  float* next_W_tile = W_tile1;

  int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

  // --- Preload first tile ---
  {
    int tile_k = 0;
    int X_col = tile_k * TILE_SIZE + threadIdx.x;  // column in X (and row in W)
    int W_row = X_col;  // Since W is (K x N)
    int X_idx = row * K + X_col;
    int W_idx = X_col * N + col;
    if (row < M && X_col < K) {
      __nv_fp4_e2m1 x_fp4 = __nv_fp4_e2m1(X[X_idx]);
      curr_X_tile[threadIdx.y * TILE_SIZE + threadIdx.x] = static_cast<float>(x_fp4);
    } else {
      curr_X_tile[threadIdx.y * TILE_SIZE + threadIdx.x] = 0.0f;
    }
    if (X_col < K && col < N) {
      __nv_fp4_e2m1 w_fp4 = __nv_fp4_e2m1(W[W_idx]);
      curr_W_tile[threadIdx.y * TILE_SIZE + threadIdx.x] = static_cast<float>(w_fp4);
    } else {
      curr_W_tile[threadIdx.y * TILE_SIZE + threadIdx.x] = 0.0f;
    }
  }
  __syncthreads();

  // --- Loop over tiles ---
  for (int t = 0; t < numTiles; t++) {
    int next_t = t + 1;
    if (next_t < numTiles) {
      int X_col_next = next_t * TILE_SIZE + threadIdx.x;
      int W_row_next = X_col_next;
      int X_idx_next = row * K + X_col_next;
      int W_idx_next = X_col_next * N + col;
      if (row < M && X_col_next < K) {
        __nv_fp4_e2m1 x_fp4 = __nv_fp4_e2m1(X[X_idx_next]);
        next_X_tile[threadIdx.y * TILE_SIZE + threadIdx.x] = static_cast<float>(x_fp4);
      } else {
        next_X_tile[threadIdx.y * TILE_SIZE + threadIdx.x] = 0.0f;
      }
      if (X_col_next < K && col < N) {
        __nv_fp4_e2m1 w_fp4 = __nv_fp4_e2m1(W[W_idx_next]);
        next_W_tile[threadIdx.y * TILE_SIZE + threadIdx.x] = static_cast<float>(w_fp4);
      } else {
        next_W_tile[threadIdx.y * TILE_SIZE + threadIdx.x] = 0.0f;
      }
    }
    __syncthreads();

    #pragma unroll
    for (int k = 0; k < TILE_SIZE; k += UNROLL_FACTOR) {
      #pragma unroll
      for (int u = 0; u < UNROLL_FACTOR; u++) {
        int idx = k + u;
        if (idx < TILE_SIZE) {
          acc += curr_X_tile[threadIdx.y * TILE_SIZE + idx] *
                 curr_W_tile[idx * TILE_SIZE + threadIdx.x];
        }
      }
    }
    __syncthreads();

    // Swap buffers for next iteration.
    float* tempX = curr_X_tile;
    float* tempW = curr_W_tile;
    curr_X_tile = next_X_tile;
    curr_W_tile = next_W_tile;
    next_X_tile = tempX;
    next_W_tile = tempW;
  }

  // Final write: perform FP4 conversion and then add bias.
  if (row < M && col < N) {
    __nv_fp4_e2m1 y_fp4 = __nv_fp4_e2m1(acc);
    float y_val = static_cast<float>(y_fp4);
    y_val += bias[col];  // Fuse bias addition.
    Y[row * N + col] = y_val;
  }
}

// -------------------------
// Host Wrapper for Fused Linear Layer
// -------------------------
torch::Tensor fp4_linear(torch::Tensor X, torch::Tensor W, torch::Tensor bias) {
  int M = X.size(0);
  int K = X.size(1);
  int N = W.size(1);  // W is assumed to be of shape (K, N)
  auto Y = torch::empty({M, N}, torch::dtype(torch::kFloat32).device(X.device()));
  dim3 threads(TILE_SIZE, TILE_SIZE);
  dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
  // Shared memory: 4 tiles for fused kernel.
  size_t shared_mem_size = 4 * TILE_SIZE * TILE_SIZE * sizeof(float);
  fp4_linear_kernel<<<blocks, threads, shared_mem_size>>>(X.data_ptr<float>(),
                                                          W.data_ptr<float>(),
                                                          bias.data_ptr<float>(),
                                                          Y.data_ptr<float>(),
                                                          M, K, N);
  return Y;
}