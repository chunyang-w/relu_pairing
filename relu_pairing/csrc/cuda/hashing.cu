#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

namespace extension_cpp {


// CUDA kernel for computing row hashes
__global__ void compute_row_hashes_kernel(const int64_t* matrix, int64_t M, int64_t N, 
  uint64_t* hashes, int64_t* row_indices) {
int row = blockIdx.x * blockDim.x + threadIdx.x;
if (row >= M) return;

// Compute polynomial hash for this row
uint64_t hash = 0;
const uint64_t base = 64;
const uint64_t mod = 1000000007ULL;  // 1e9 + 7

for (int64_t col = 0; col < N; col++) {
int64_t val = matrix[row * N + col];
hash = (hash * base + val) % mod;
}

hashes[row] = hash;
row_indices[row] = row;
}

// CUDA kernel for finding pairs from sorted hash-index pairs with row equality verification
__global__ void find_pairs_kernel(const uint64_t* sorted_hashes, const int64_t* sorted_indices, 
                                 const int64_t* matrix, int64_t M, int64_t N, 
                                 int64_t* pairs, int64_t* pair_count, int64_t* used_flags) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= M) return;

  // Skip if this row is already used
  if (used_flags[idx] == 1) return;

  int64_t row1 = sorted_indices[idx];
  
  // Look for a matching row among the remaining rows with the same hash
  for (int64_t search_idx = idx + 1; search_idx < M; search_idx++) {
    // Stop if hash changes
    if (sorted_hashes[search_idx] != sorted_hashes[idx]) break;
    
    // Skip if this candidate row is already used
    if (used_flags[search_idx] == 1) continue;
    
    int64_t row2 = sorted_indices[search_idx];
    
    // Check if rows are actually equal
    bool rows_equal = true;
    for (int64_t col = 0; col < N; col++) {
      if (matrix[row1 * N + col] != matrix[row2 * N + col]) {
        rows_equal = false;
        break;
      }
    }
    
    // If rows are equal, try to pair them
    if (rows_equal) {
      // Try to atomically mark both rows as used
      if (atomicCAS((unsigned long long*)&used_flags[idx], 0ULL, 1ULL) == 0ULL) {
        if (atomicCAS((unsigned long long*)&used_flags[search_idx], 0ULL, 1ULL) == 0ULL) {
          // Successfully marked both as used, create the pair
          int64_t pair_idx = atomicAdd((unsigned long long*)pair_count, 1ULL);
          if (pair_idx < M / 2) {
            pairs[pair_idx * 2] = row1;
            pairs[pair_idx * 2 + 1] = row2;
          }
          break;
        } else {
          // Failed to mark second row, unmark first row
          used_flags[idx] = 0;
        }
      }
      // If we failed to mark the first row, someone else got it, so we're done
      if (used_flags[idx] == 1) break;
    }
  }
}

// CUDA kernel for computing hashes (Phase 1)
__global__ void compute_hashes_kernel(const int64_t* matrix, int64_t M, int64_t N, uint64_t* hashes) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M) return;

  // Compute polynomial hash for this row
  uint64_t hash = 0;
  const uint64_t base = 64;
  const uint64_t mod = 1000000007ULL;  // 1e9 + 7

  for (int64_t col = 0; col < N; col++) {
    int64_t val = matrix[row * N + col];
    hash = (hash * base + val) % mod;
  }

  hashes[row] = hash;
}

// CUDA kernel for simple O(N) pairing - each thread looks for its pair in all other rows
__global__ void find_pairs_simple_kernel(const int64_t* matrix, const uint64_t* hashes, 
                                         int64_t M, int64_t N, int64_t* pairs, 
                                         int64_t* pair_count, int64_t* used_flags) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M) return;

  // Skip if this row is already paired
  if (used_flags[row] == 1) return;
  
  uint64_t my_hash = hashes[row];
  
  // Look for a matching row with higher index (to avoid duplicate pairs)
  for (int64_t other_row = row + 1; other_row < M; other_row++) {
    // Skip if other row is already used
    if (used_flags[other_row] == 1) continue;
    
    // Quick hash check first
    if (hashes[other_row] != my_hash) continue;
    
    // Hash matches - verify actual row equality
    bool rows_equal = true;
    for (int64_t col = 0; col < N; col++) {
      if (matrix[row * N + col] != matrix[other_row * N + col]) {
        rows_equal = false;
        break;
      }
    }
    
    if (rows_equal) {
      // Found a match! Try to atomically claim both rows
      if (atomicCAS((unsigned long long*)&used_flags[row], 0ULL, 1ULL) == 0ULL) {
        if (atomicCAS((unsigned long long*)&used_flags[other_row], 0ULL, 1ULL) == 0ULL) {
          // Successfully claimed both rows
          int64_t pair_idx = atomicAdd((unsigned long long*)pair_count, 1ULL);
          if (pair_idx < M / 2) {
            pairs[pair_idx * 2] = row;
            pairs[pair_idx * 2 + 1] = other_row;
          }
          break;  // Found our pair, done
        } else {
          // Failed to claim other row, release our row
          used_flags[row] = 0;
        }
      }
      // If we couldn't claim our row, someone else got it, so we're done
      if (used_flags[row] == 1) break;
    }
  }
}

at::Tensor hash_pair_rows_cuda(const at::Tensor& matrix) {
  TORCH_CHECK(matrix.dim() == 2, "Input must be a 2D matrix");
  TORCH_CHECK(matrix.dtype() == at::kLong, "Matrix must be of integer type");
TORCH_INTERNAL_ASSERT(matrix.device().type() == at::DeviceType::CUDA);

int64_t M = matrix.size(0);  // number of rows
int64_t N = matrix.size(1);  // number of columns

TORCH_CHECK(M % 2 == 0, "Number of rows must be even for complete pairing");

at::Tensor matrix_contig = matrix.contiguous();
const int64_t* matrix_ptr = matrix_contig.data_ptr<int64_t>();

// Create result tensor for pairs: shape (M/2, 2)
at::Tensor pairs = torch::zeros({M / 2, 2}, 
matrix.options().dtype(at::kLong));
int64_t* pairs_ptr = pairs.data_ptr<int64_t>();

// Allocate temporary arrays for hashing
thrust::device_vector<uint64_t> hashes(M);
thrust::device_vector<int64_t> row_indices(M);
thrust::device_vector<int64_t> pair_count_vec(1, 0);
thrust::device_vector<int64_t> used_flags(M, 0);  // Track which rows are already paired

uint64_t* hashes_ptr = thrust::raw_pointer_cast(hashes.data());
int64_t* indices_ptr = thrust::raw_pointer_cast(row_indices.data());
int64_t* pair_count_ptr = thrust::raw_pointer_cast(pair_count_vec.data());
int64_t* used_flags_ptr = thrust::raw_pointer_cast(used_flags.data());

// Compute hashes for all rows
int threads_per_block = 256;
int blocks = (M + threads_per_block - 1) / threads_per_block;
compute_row_hashes_kernel<<<blocks, threads_per_block>>>(
matrix_ptr, M, N, hashes_ptr, indices_ptr);
cudaDeviceSynchronize();

// Sort by hash value while keeping track of original indices
thrust::sort_by_key(hashes.begin(), hashes.end(), row_indices.begin());

// Find pairs from sorted hash-index pairs
blocks = (M + threads_per_block - 1) / threads_per_block;
find_pairs_kernel<<<blocks, threads_per_block>>>(
hashes_ptr, indices_ptr, matrix_ptr, M, N, pairs_ptr, pair_count_ptr, used_flags_ptr);
cudaDeviceSynchronize();

// Verify we found the expected number of pairs
int64_t found_pairs = pair_count_vec[0];
TORCH_CHECK(found_pairs == M / 2, 
"Could not find complete pairing - some rows may be unique. "
"Found ", found_pairs, " pairs, expected ", M / 2);

return pairs;
}

at::Tensor hash_pair_rows_cuda_o_n(const at::Tensor& matrix) {
  TORCH_CHECK(matrix.dim() == 2, "Input must be a 2D matrix");
  TORCH_CHECK(matrix.dtype() == at::kLong, "Matrix must be of integer type");
  TORCH_INTERNAL_ASSERT(matrix.device().type() == at::DeviceType::CUDA);
  
  int64_t M = matrix.size(0);  // number of rows
  int64_t N = matrix.size(1);  // number of columns
  
  TORCH_CHECK(M % 2 == 0, "Number of rows must be even for complete pairing");
  
  at::Tensor matrix_contig = matrix.contiguous();
  const int64_t* matrix_ptr = matrix_contig.data_ptr<int64_t>();
  
  // Create result tensor for pairs: shape (M/2, 2)
  at::Tensor pairs = torch::zeros({M / 2, 2}, 
                                  matrix.options().dtype(at::kLong));
  int64_t* pairs_ptr = pairs.data_ptr<int64_t>();
  
  // Phase 1: Compute all hashes first
  thrust::device_vector<uint64_t> hashes(M);
  uint64_t* hashes_ptr = thrust::raw_pointer_cast(hashes.data());
  
  int threads_per_block = 256;
  int blocks = (M + threads_per_block - 1) / threads_per_block;
  
  compute_hashes_kernel<<<blocks, threads_per_block>>>(matrix_ptr, M, N, hashes_ptr);
  cudaDeviceSynchronize();  // Wait for all hashes to be computed
  
  // Phase 2: Find pairs using pre-computed hashes
  // Create used flags array to track which rows are already paired
  thrust::device_vector<int64_t> used_flags(M, 0);
  thrust::device_vector<int64_t> pair_count_vec(1, 0);
  
  int64_t* used_flags_ptr = thrust::raw_pointer_cast(used_flags.data());
  int64_t* pair_count_ptr = thrust::raw_pointer_cast(pair_count_vec.data());
  
  // Launch pairing kernel - each thread processes one row
  find_pairs_simple_kernel<<<blocks, threads_per_block>>>(
    matrix_ptr, hashes_ptr, M, N, pairs_ptr, pair_count_ptr, used_flags_ptr);
  cudaDeviceSynchronize();
  
  // Verify we found the expected number of pairs
  int64_t found_pairs = pair_count_vec[0];
  TORCH_CHECK(found_pairs == M / 2, 
              "Could not find complete pairing - some rows may be unique. "
              "Found ", found_pairs, " pairs, expected ", M / 2);
  
  return pairs;
}


// Register the O(N) simple CUDA implementation
TORCH_LIBRARY_IMPL(extension_cpp, CUDA, m) {
  m.impl("hash_pair_rows_simple", &hash_pair_rows_cuda_o_n);
}

}
