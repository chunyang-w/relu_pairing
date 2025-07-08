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
const uint64_t base = 31;
const uint64_t mod = 1000000007ULL;  // 1e9 + 7

for (int64_t col = 0; col < N; col++) {
int64_t val = matrix[row * N + col];
hash = (hash * base + val) % mod;
}

hashes[row] = hash;
row_indices[row] = row;
}

// CUDA kernel for finding pairs from sorted hash-index pairs
__global__ void find_pairs_kernel(const uint64_t* sorted_hashes, 
const int64_t* sorted_indices,
int64_t M, int64_t* pairs, int64_t* pair_count) {
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx >= M - 1) return;

// Check if current and next hash are the same
if (sorted_hashes[idx] == sorted_hashes[idx + 1]) {
// Atomic increment to get unique pair index
int64_t pair_idx = atomicAdd((unsigned long long*)pair_count, 1ULL);
if (pair_idx < M / 2) {
pairs[pair_idx * 2] = sorted_indices[idx];
pairs[pair_idx * 2 + 1] = sorted_indices[idx + 1];
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

uint64_t* hashes_ptr = thrust::raw_pointer_cast(hashes.data());
int64_t* indices_ptr = thrust::raw_pointer_cast(row_indices.data());
int64_t* pair_count_ptr = thrust::raw_pointer_cast(pair_count_vec.data());

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
hashes_ptr, indices_ptr, M, pairs_ptr, pair_count_ptr);
cudaDeviceSynchronize();

// Verify we found the expected number of pairs
int64_t found_pairs = pair_count_vec[0];
TORCH_CHECK(found_pairs == M / 2, 
"Could not find complete pairing - some rows may be unique. "
"Found ", found_pairs, " pairs, expected ", M / 2);

return pairs;
}

__global__ void muladd_kernel(int numel, const float* a, const float* b, float c, float* result) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numel) result[idx] = a[idx] * b[idx] + c;
}

at::Tensor mymuladd_cuda(const at::Tensor& a, const at::Tensor& b, double c) {
  TORCH_CHECK(a.sizes() == b.sizes());
  TORCH_CHECK(a.dtype() == at::kFloat);
  TORCH_CHECK(b.dtype() == at::kFloat);
  TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CUDA);
  TORCH_INTERNAL_ASSERT(b.device().type() == at::DeviceType::CUDA);
  at::Tensor a_contig = a.contiguous();
  at::Tensor b_contig = b.contiguous();
  at::Tensor result = at::empty(a_contig.sizes(), a_contig.options());
  const float* a_ptr = a_contig.data_ptr<float>();
  const float* b_ptr = b_contig.data_ptr<float>();
  float* result_ptr = result.data_ptr<float>();

  int numel = a_contig.numel();
  muladd_kernel<<<(numel+255)/256, 256>>>(numel, a_ptr, b_ptr, c, result_ptr);
  return result;
}

__global__ void mul_kernel(int numel, const float* a, const float* b, float* result) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numel) result[idx] = a[idx] * b[idx];
}

at::Tensor mymul_cuda(const at::Tensor& a, const at::Tensor& b) {
  TORCH_CHECK(a.sizes() == b.sizes());
  TORCH_CHECK(a.dtype() == at::kFloat);
  TORCH_CHECK(b.dtype() == at::kFloat);
  TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CUDA);
  TORCH_INTERNAL_ASSERT(b.device().type() == at::DeviceType::CUDA);
  at::Tensor a_contig = a.contiguous();
  at::Tensor b_contig = b.contiguous();
  at::Tensor result = at::empty(a_contig.sizes(), a_contig.options());
  const float* a_ptr = a_contig.data_ptr<float>();
  const float* b_ptr = b_contig.data_ptr<float>();
  float* result_ptr = result.data_ptr<float>();
  int numel = a_contig.numel();
  mul_kernel<<<(numel+255)/256, 256>>>(numel, a_ptr, b_ptr, result_ptr);
  return result;
}

__global__ void add_kernel(int numel, const float* a, const float* b, float* result) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numel) result[idx] = a[idx] + b[idx];
}

void myadd_out_cuda(const at::Tensor& a, const at::Tensor& b, at::Tensor& out) {
  TORCH_CHECK(a.sizes() == b.sizes());
  TORCH_CHECK(b.sizes() == out.sizes());
  TORCH_CHECK(a.dtype() == at::kFloat);
  TORCH_CHECK(b.dtype() == at::kFloat);
  TORCH_CHECK(out.dtype() == at::kFloat);
  TORCH_CHECK(out.is_contiguous());
  TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CUDA);
  TORCH_INTERNAL_ASSERT(b.device().type() == at::DeviceType::CUDA);
  TORCH_INTERNAL_ASSERT(out.device().type() == at::DeviceType::CUDA);
  at::Tensor a_contig = a.contiguous();
  at::Tensor b_contig = b.contiguous();
  const float* a_ptr = a_contig.data_ptr<float>();
  const float* b_ptr = b_contig.data_ptr<float>();
  float* result_ptr = out.data_ptr<float>();
  int numel = a_contig.numel();
  add_kernel<<<(numel+255)/256, 256>>>(numel, a_ptr, b_ptr, result_ptr);
}


// Registers CUDA implementations for mymuladd, mymul, myadd_out
TORCH_LIBRARY_IMPL(extension_cpp, CUDA, m) {
  m.impl("mymuladd", &mymuladd_cuda);
  m.impl("mymul", &mymul_cuda);
  m.impl("myadd_out", &myadd_out_cuda);
  m.impl("hash_pair_rows", &hash_pair_rows_cuda);
}

}
