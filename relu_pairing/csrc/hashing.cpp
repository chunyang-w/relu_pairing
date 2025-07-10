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
  
  // Hash map to store row hash -> vector of row indices (for collision handling)
  std::unordered_map<uint64_t, std::vector<int64_t>> row_hash_map;
  int64_t pair_count = 0;
  
  // Helper function to check if two rows are actually equal
  auto rows_equal = [&](int64_t row1, int64_t row2) -> bool {
    for (int64_t col = 0; col < N; col++) {
      if (matrix_ptr[row1 * N + col] != matrix_ptr[row2 * N + col]) {
        return false;
      }
    }
    return true;
  };
  
  // Process each row
  for (int64_t row = 0; row < M; row++) {
    // Compute hash for current row using polynomial rolling hash
    uint64_t hash = 0;
    const uint64_t base = 64;
    const uint64_t mod = 1e9 + 7;
    
    for (int64_t col = 0; col < N; col++) {
      int64_t val = matrix_ptr[row * N + col];
      hash = (hash * base + val) % mod;
    }
    
    // Check if we've seen this hash before
    auto it = row_hash_map.find(hash);
    if (it != row_hash_map.end()) {
      // Hash collision detected - need to check actual row equality
      bool found_match = false;
      auto& candidate_rows = it->second;
      
      // Check each candidate row with the same hash
      for (auto candidate_it = candidate_rows.begin(); 
           candidate_it != candidate_rows.end(); ++candidate_it) {
        if (rows_equal(*candidate_it, row)) {
          // Found actual row match! Create a pair
          pairs_ptr[pair_count * 2] = *candidate_it;  // first row index
          pairs_ptr[pair_count * 2 + 1] = row;        // second row index
          pair_count++;
          found_match = true;
          
          // Remove the matched row from candidates
          candidate_rows.erase(candidate_it);
          
          // If no more candidates for this hash, remove hash entry
          if (candidate_rows.empty()) {
            row_hash_map.erase(it);
          }
          break;
        }
      }
      
      if (!found_match) {
        // Hash collision but no actual match - add to candidates
        candidate_rows.push_back(row);
      }
    } else {
      // First time seeing this hash, create new entry
      row_hash_map[hash] = std::vector<int64_t>{row};
    }
  }
  
  TORCH_CHECK(pair_count == M / 2, 
              "Could not find complete pairing - some rows may be unique");
  
  return pairs;
}


// Defines the operators
TORCH_LIBRARY(extension_cpp, m) {
  m.def("hash_pair_rows(Tensor matrix) -> Tensor");
  m.def("hash_pair_rows_sorted(Tensor matrix) -> Tensor");
  m.def("hash_pair_rows_simple(Tensor matrix) -> Tensor");
}

// Register CPU implementations
TORCH_LIBRARY_IMPL(extension_cpp, CPU, m) {
  m.impl("hash_pair_rows", &hash_pair_rows_cpu);
  // For CPU, we only have the main hash_pair_rows method
  // The CUDA-specific methods will fallback to CPU if needed
}

}
