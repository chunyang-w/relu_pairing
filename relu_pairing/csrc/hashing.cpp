#include <Python.h>
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

#include <vector>
#include <unordered_map>

extern "C" {
  /* Creates a dummy empty _C module that can be imported from Python.
     The import from Python will load the .so consisting of this file
     in this extension, so that the TORCH_LIBRARY static initializers
     below are run. */
  PyObject* PyInit__C(void)
  {
      static struct PyModuleDef module_def = {
          PyModuleDef_HEAD_INIT,
          "_C",   /* name of module */
          NULL,   /* module documentation, may be NULL */
          -1,     /* size of per-interpreter state of the module,
                     or -1 if the module keeps state in global variables. */
          NULL,   /* methods */
      };
      return PyModule_Create(&module_def);
  }
}

namespace extension_cpp {


// Hash-based row pairing function
at::Tensor hash_pair_rows_cpu(const at::Tensor& matrix) {
  TORCH_CHECK(matrix.dim() == 2, "Input must be a 2D matrix");
  TORCH_CHECK(matrix.dtype() == at::kLong, "Matrix must be of integer type");
  TORCH_INTERNAL_ASSERT(matrix.device().type() == at::DeviceType::CPU);
  
  int64_t M = matrix.size(0);  // number of rows
  int64_t N = matrix.size(1);  // number of columns
  
  TORCH_CHECK(M % 2 == 0, "Number of rows must be even for complete pairing");
  
  at::Tensor matrix_contig = matrix.contiguous();
  const int64_t* matrix_ptr = matrix_contig.data_ptr<int64_t>();
  
  // Create result tensor for pairs: shape (M/2, 2)
  at::Tensor pairs = torch::empty({M / 2, 2}, 
                                  matrix.options().dtype(at::kLong));
  int64_t* pairs_ptr = pairs.data_ptr<int64_t>();
  
  // Hash map to store row hash -> row index
  std::unordered_map<uint64_t, int64_t> row_hash_map;
  int64_t pair_count = 0;
  
  // Process each row
  for (int64_t row = 0; row < M; row++) {
    // Compute hash for current row using polynomial rolling hash
    uint64_t hash = 0;
    const uint64_t base = 31;
    const uint64_t mod = 1e9 + 7;
    
    for (int64_t col = 0; col < N; col++) {
      int64_t val = matrix_ptr[row * N + col];
      hash = (hash * base + val) % mod;
    }
    
    // Check if we've seen this hash before
    auto it = row_hash_map.find(hash);
    if (it != row_hash_map.end()) {
      // Found a match! Create a pair
      pairs_ptr[pair_count * 2] = it->second;     // first row index
      pairs_ptr[pair_count * 2 + 1] = row;        // second row index
      pair_count++;
      
      // Remove the hash from map since we've paired it
      row_hash_map.erase(it);
    } else {
      // First time seeing this hash, store it
      row_hash_map[hash] = row;
    }
  }
  
  TORCH_CHECK(pair_count == M / 2, 
              "Could not find complete pairing - some rows may be unique");
  
  return pairs;
}

at::Tensor mymuladd_cpu(const at::Tensor& a, const at::Tensor& b, double c) {
  TORCH_CHECK(a.sizes() == b.sizes());
  TORCH_CHECK(a.dtype() == at::kFloat);
  TORCH_CHECK(b.dtype() == at::kFloat);
  TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CPU);
  TORCH_INTERNAL_ASSERT(b.device().type() == at::DeviceType::CPU);
  at::Tensor a_contig = a.contiguous();
  at::Tensor b_contig = b.contiguous();
  at::Tensor result = torch::empty(a_contig.sizes(), a_contig.options());
  const float* a_ptr = a_contig.data_ptr<float>();
  const float* b_ptr = b_contig.data_ptr<float>();
  float* result_ptr = result.data_ptr<float>();
  for (int64_t i = 0; i < result.numel(); i++) {
    result_ptr[i] = a_ptr[i] * b_ptr[i] + c;
  }
  return result;
}

at::Tensor mymul_cpu(const at::Tensor& a, const at::Tensor& b) {
  TORCH_CHECK(a.sizes() == b.sizes());
  TORCH_CHECK(a.dtype() == at::kFloat);
  TORCH_CHECK(b.dtype() == at::kFloat);
  TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CPU);
  TORCH_INTERNAL_ASSERT(b.device().type() == at::DeviceType::CPU);
  at::Tensor a_contig = a.contiguous();
  at::Tensor b_contig = b.contiguous();
  at::Tensor result = torch::empty(a_contig.sizes(), a_contig.options());
  const float* a_ptr = a_contig.data_ptr<float>();
  const float* b_ptr = b_contig.data_ptr<float>();
  float* result_ptr = result.data_ptr<float>();
  for (int64_t i = 0; i < result.numel(); i++) {
    result_ptr[i] = a_ptr[i] * b_ptr[i];
  }
  return result;
}

// An example of an operator that mutates one of its inputs.
void myadd_out_cpu(const at::Tensor& a, const at::Tensor& b, at::Tensor& out) {
  TORCH_CHECK(a.sizes() == b.sizes());
  TORCH_CHECK(b.sizes() == out.sizes());
  TORCH_CHECK(a.dtype() == at::kFloat);
  TORCH_CHECK(b.dtype() == at::kFloat);
  TORCH_CHECK(out.dtype() == at::kFloat);
  TORCH_CHECK(out.is_contiguous());
  TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CPU);
  TORCH_INTERNAL_ASSERT(b.device().type() == at::DeviceType::CPU);
  TORCH_INTERNAL_ASSERT(out.device().type() == at::DeviceType::CPU);
  at::Tensor a_contig = a.contiguous();
  at::Tensor b_contig = b.contiguous();
  const float* a_ptr = a_contig.data_ptr<float>();
  const float* b_ptr = b_contig.data_ptr<float>();
  float* result_ptr = out.data_ptr<float>();
  for (int64_t i = 0; i < out.numel(); i++) {
    result_ptr[i] = a_ptr[i] + b_ptr[i];
  }
}

// Defines the operators
TORCH_LIBRARY(extension_cpp, m) {
  m.def("mymuladd(Tensor a, Tensor b, float c) -> Tensor");
  m.def("mymul(Tensor a, Tensor b) -> Tensor");
  m.def("myadd_out(Tensor a, Tensor b, Tensor(a!) out) -> ()");
  m.def("hash_pair_rows(Tensor matrix) -> Tensor");
}

// Registers CUDA implementations for mymuladd, mymul, myadd_out
TORCH_LIBRARY_IMPL(extension_cpp, CPU, m) {
  m.impl("mymuladd", &mymuladd_cpu);
  m.impl("mymul", &mymul_cpu);
  m.impl("myadd_out", &myadd_out_cpu);
  m.impl("hash_pair_rows", &hash_pair_rows_cpu);
}

}
