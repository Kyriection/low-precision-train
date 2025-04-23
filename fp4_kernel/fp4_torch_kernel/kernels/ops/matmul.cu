#include <cuda_fp4.h>
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_SIZE 128
#define UNROLL_FACTOR 8

// -------------------------
// Forward Kernel
// -------------------------
__global__ void fp4_matmul_kernel(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C,
                                  int M, int K, int N) {
  const int row = blockIdx.y * TILE_SIZE + threadIdx.y;
  const int col = blockIdx.x * TILE_SIZE + threadIdx.x;
  float acc = 0.0f;

  // Allocate shared memory for two buffers:
  // Layout: A_tile0, B_tile0, A_tile1, B_tile1 (each TILE_SIZE*TILE_SIZE floats)
  extern __shared__ float shared_mem[];
  float* A_tile0 = shared_mem;
  float* B_tile0 = A_tile0 + TILE_SIZE * TILE_SIZE;
  float* A_tile1 = B_tile0 + TILE_SIZE * TILE_SIZE;
  float* B_tile1 = A_tile1 + TILE_SIZE * TILE_SIZE;

  // Pointers to current and next buffers.
  float* curr_A_tile = A_tile0;
  float* curr_B_tile = B_tile0;
  float* next_A_tile = A_tile1;
  float* next_B_tile = B_tile1;

  int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

  // --- Preload first tile ---
  {
    const int A_col = 0 * TILE_SIZE + threadIdx.x;
    const int B_row = 0 * TILE_SIZE + threadIdx.y;
    int A_idx = row * K + A_col;
    int B_idx = B_row * N + col;
    if (row < M && A_col < K) {
      __nv_fp4_e2m1 a_fp4 = __nv_fp4_e2m1(A[A_idx]);
      curr_A_tile[threadIdx.y * TILE_SIZE + threadIdx.x] = static_cast<float>(a_fp4);
    } else {
      curr_A_tile[threadIdx.y * TILE_SIZE + threadIdx.x] = 0.0f;
    }
    if (B_row < K && col < N) {
      __nv_fp4_e2m1 b_fp4 = __nv_fp4_e2m1(B[B_idx]);
      curr_B_tile[threadIdx.y * TILE_SIZE + threadIdx.x] = static_cast<float>(b_fp4);
    } else {
      curr_B_tile[threadIdx.y * TILE_SIZE + threadIdx.x] = 0.0f;
    }
  }
  __syncthreads();

  // --- Loop over tiles ---
  for (int t = 0; t < numTiles; t++) {
    const int next_t = t + 1;
    if (next_t < numTiles) {
      const int A_col_next = next_t * TILE_SIZE + threadIdx.x;
      const int B_row_next = next_t * TILE_SIZE + threadIdx.y;
      int A_idx_next = row * K + A_col_next;
      int B_idx_next = B_row_next * N + col;
      if (row < M && A_col_next < K) {
        __nv_fp4_e2m1 a_fp4 = __nv_fp4_e2m1(A[A_idx_next]);
        next_A_tile[threadIdx.y * TILE_SIZE + threadIdx.x] = static_cast<float>(a_fp4);
      } else {
        next_A_tile[threadIdx.y * TILE_SIZE + threadIdx.x] = 0.0f;
      }
      if (B_row_next < K && col < N) {
        __nv_fp4_e2m1 b_fp4 = __nv_fp4_e2m1(B[B_idx_next]);
        next_B_tile[threadIdx.y * TILE_SIZE + threadIdx.x] = static_cast<float>(b_fp4);
      } else {
        next_B_tile[threadIdx.y * TILE_SIZE + threadIdx.x] = 0.0f;
      }
    }

    __syncthreads();

    #pragma unroll
    for (int k = 0; k < TILE_SIZE; k += UNROLL_FACTOR) {
      #pragma unroll
      for (int u = 0; u < UNROLL_FACTOR; u++) {
        int idx = k + u;
        if (idx < TILE_SIZE) {
          acc += curr_A_tile[threadIdx.y * TILE_SIZE + idx] *
                 curr_B_tile[idx * TILE_SIZE + threadIdx.x];
        }
      }
    }
    __syncthreads();

    // Swap buffers.
    float* tempA = curr_A_tile;
    float* tempB = curr_B_tile;
    curr_A_tile = next_A_tile;
    curr_B_tile = next_B_tile;
    next_A_tile = tempA;
    next_B_tile = tempB;
  }

  if (row < M && col < N) {
    __nv_fp4_e2m1 c_fp4 = __nv_fp4_e2m1(acc);
    C[row * N + col] = static_cast<float>(c_fp4);
  }
}

// -------------------------
// Backward Kernel for grad_A = grad_output * B^T
// -------------------------
//
// For each element (row, col) in grad_A (shape: M x K):
//   grad_A[row, col] = sum_{j=0}^{N-1} grad_output[row, j] * B[col, j]
// We tile along the j (i.e. N) dimension. To improve memory coalescing, we load B in transposed form.
__global__ void fp4_matmul_backward_A_kernel(const float* __restrict__ grad_output,
                                               const float* __restrict__ B,
                                               float* __restrict__ grad_A,
                                               int M, int N, int K) {
  const int row = blockIdx.y * TILE_SIZE + threadIdx.y;  // index in grad_output / grad_A
  const int col = blockIdx.x * TILE_SIZE + threadIdx.x;  // index in grad_A (and row in B)
  float acc = 0.0f;

  // Allocate shared memory:
  // grad_tile: tile of grad_output (size TILE_SIZE x TILE_SIZE)
  // B_tile: tile of B loaded in transposed form (size TILE_SIZE x TILE_SIZE)
  extern __shared__ float shared_mem[];
  float* grad_tile = shared_mem;
  float* B_tile = shared_mem + TILE_SIZE * TILE_SIZE;

  int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;
  for (int t = 0; t < numTiles; t++) {
    int tile_j = t * TILE_SIZE + threadIdx.x;
    // Load grad_output tile element: grad_output[row, t*TILE_SIZE + threadIdx.x]
    if (row < M && tile_j < N) {
      grad_tile[threadIdx.y * TILE_SIZE + threadIdx.x] = grad_output[row * N + tile_j];
    } else {
      grad_tile[threadIdx.y * TILE_SIZE + threadIdx.x] = 0.0f;
    }

    int tile_j_B = t * TILE_SIZE + threadIdx.y;  // use threadIdx.y for loading B so we can transpose
    if (col < K && tile_j_B < N) {
      B_tile[threadIdx.x * TILE_SIZE + threadIdx.y] = B[col * N + tile_j_B];
    } else {
      B_tile[threadIdx.x * TILE_SIZE + threadIdx.y] = 0.0f;
    }

    __syncthreads();

    // Multiply tile: dot product over the tile dimension.
    #pragma unroll
    for (int j = 0; j < TILE_SIZE; j += UNROLL_FACTOR) {
      #pragma unroll
      for (int u = 0; u < UNROLL_FACTOR; u++) {
        int idx = j + u;
        if (idx < TILE_SIZE) {
          acc += grad_tile[threadIdx.y * TILE_SIZE + idx] *
                 B_tile[idx * TILE_SIZE + threadIdx.x];
        }
      }
    }
    __syncthreads();
  }
  if (row < M && col < K) {
    __nv_fp4_e2m1 res_fp4 = __nv_fp4_e2m1(acc);
    grad_A[row * K + col] = static_cast<float>(res_fp4);
  }
}

// -------------------------
// Backward Kernel for grad_B = A^T * grad_output
// -------------------------
//
// For each element (row, col) in grad_B (shape: K x N):
//   grad_B[row, col] = sum_{i=0}^{M-1} A[i, row] * grad_output[i, col]
// We tile along the i (i.e. M) dimension. We load A in transposed form.
__global__ void fp4_matmul_backward_B_kernel(const float* __restrict__ A,
                                               const float* __restrict__ grad_output,
                                               float* __restrict__ grad_B,
                                               int M, int N, int K) {
  const int row = blockIdx.y * TILE_SIZE + threadIdx.y;  // index in grad_B (and column in A)
  const int col = blockIdx.x * TILE_SIZE + threadIdx.x;  // index in grad_B and grad_output
  float acc = 0.0f;

  // Allocate shared memory:
  // A_tile: tile of A loaded in transposed form (size TILE_SIZE x TILE_SIZE)
  // grad_tile: tile of grad_output (size TILE_SIZE x TILE_SIZE)
  extern __shared__ float shared_mem[];
  float* A_tile = shared_mem;
  float* grad_tile = shared_mem + TILE_SIZE * TILE_SIZE;

  int numTiles = (M + TILE_SIZE - 1) / TILE_SIZE;
  for (int t = 0; t < numTiles; t++) {
    int tile_i = t * TILE_SIZE + threadIdx.x;
    // Load A tile transposed: element A[tile_i, row] (A is MxK)
    if (row < K && tile_i < M) {
      A_tile[threadIdx.x * TILE_SIZE + threadIdx.y] = A[tile_i * K + row];
    } else {
      A_tile[threadIdx.x * TILE_SIZE + threadIdx.y] = 0.0f;
    }

    int tile_i_grad = t * TILE_SIZE + threadIdx.y;
    // Load grad_output tile: element grad_output[tile_i_grad, col]
    if (col < N && tile_i_grad < M) {
      grad_tile[threadIdx.x * TILE_SIZE + threadIdx.y] = grad_output[tile_i_grad * N + col];
    } else {
      grad_tile[threadIdx.x * TILE_SIZE + threadIdx.y] = 0.0f;
    }

    __syncthreads();

    #pragma unroll
    for (int i = 0; i < TILE_SIZE; i += UNROLL_FACTOR) {
      #pragma unroll
      for (int u = 0; u < UNROLL_FACTOR; u++) {
        int idx = i + u;
        if (idx < TILE_SIZE) {
          acc += A_tile[idx * TILE_SIZE + threadIdx.y] *
                 grad_tile[idx * TILE_SIZE + threadIdx.x];
        }
      }
    }
    __syncthreads();
  }
  if (row < K && col < N) {
    __nv_fp4_e2m1 res_fp4 = __nv_fp4_e2m1(acc);
    grad_B[row * N + col] = static_cast<float>(res_fp4);
  }
}

// -------------------------
// Host Wrappers
// -------------------------
torch::Tensor fp4_matmul(torch::Tensor A, torch::Tensor B) {
  int M = A.size(0);
  int K = A.size(1);
  int N = B.size(1);
  auto C = torch::empty({M, N}, torch::dtype(torch::kFloat32).device(A.device()));
  dim3 threads(TILE_SIZE, TILE_SIZE);
  dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
  // Shared memory: 4 tiles for forward kernel.
  size_t shared_mem_size = 4 * TILE_SIZE * TILE_SIZE * sizeof(float);
  fp4_matmul_kernel<<<blocks, threads, shared_mem_size>>>(
      A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, K, N);
  return C;
}

torch::Tensor fp4_matmul_backward_A(torch::Tensor grad_output, torch::Tensor B) {
  int M = grad_output.size(0);
  int N = grad_output.size(1);
  int K = B.size(0);  // B is (K x N)
  auto grad_A = torch::empty({M, K}, torch::dtype(torch::kFloat32).device(grad_output.device()));
  dim3 threads(TILE_SIZE, TILE_SIZE);
  dim3 blocks((K + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
  // Shared memory: two tiles (grad_tile and B_tile)
  size_t shared_mem_size = 2 * TILE_SIZE * TILE_SIZE * sizeof(float);
  fp4_matmul_backward_A_kernel<<<blocks, threads, shared_mem_size>>>(
      grad_output.data_ptr<float>(), B.data_ptr<float>(), grad_A.data_ptr<float>(), M, N, K);
  return grad_A;
}

torch::Tensor fp4_matmul_backward_B(torch::Tensor A, torch::Tensor grad_output) {
  int M = A.size(0);
  int K = A.size(1);
  int N = grad_output.size(1);
  auto grad_B = torch::empty({K, N}, torch::dtype(torch::kFloat32).device(A.device()));
  dim3 threads(TILE_SIZE, TILE_SIZE);
  dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (K + TILE_SIZE - 1) / TILE_SIZE);
  // Shared memory: two tiles (A_tile and grad_tile)
  size_t shared_mem_size = 2 * TILE_SIZE * TILE_SIZE * sizeof(float);
  fp4_matmul_backward_B_kernel<<<blocks, threads, shared_mem_size>>>(
      A.data_ptr<float>(), grad_output.data_ptr<float>(), grad_B.data_ptr<float>(), M, N, K);
  return grad_B;
}