# C++/CUDA Extensions in PyTorch

Pairing M rows in a Matrix of 2M by N - hashing version.

The examples in this repo work with PyTorch 2.4+.

```
===============🧪 SYNTHETIC LARGE MATRIX TEST================
Creating synthetic matrix of shape 1000000x2000...

        Large Matrix Benchmarks         

==================== BENCHMARKING ON CPU ====================
Matrix shape: torch.Size([1000000, 2000])
Expected pairs: 500000

--------------- Hash-based Method (CPU) ---------------
✅ Hash CPU method: 7.8524s, found 500000 pairs

--------------- Sort-based Method (Ground Truth) ---------------
✅ Sort method: 91.6134s, found 500000 pairs

--------------- Naive Method (Skipped - too large) ---------------

--------------- CORRECTNESS VERIFICATION ---------------
hash_cpu: ✓ All pairs correct
sort: ✓ All pairs correct

--------------- PERFORMANCE ANALYSIS ---------------
🏆 Fastest method: hash_cpu (7.8524s)

Speedup vs sort baseline (91.6134s):
  hash_cpu: 11.67x faster

==================== BENCHMARKING ON CUDA ====================
Matrix shape: torch.Size([1000000, 2000])
Expected pairs: 500000

--------------- Hash-based Method (CPU) ---------------
✅ Hash CPU method: 11.4183s, found 500000 pairs

--------------- Hash-based Method (CUDA Sorted) ---------------
✅ Hash CUDA Sorted method: 0.0879s, found 500000 pairs

--------------- Hash-based Method (CUDA Simple O(N)) ---------------
✅ Hash CUDA Simple method: 0.8640s, found 500000 pairs

--------------- Sort-based Method (Ground Truth) ---------------
✅ Sort method: 95.2078s, found 500000 pairs

--------------- Naive Method (Skipped - too large) ---------------

--------------- CORRECTNESS VERIFICATION ---------------
hash_cpu: ✓ All pairs correct
hash_sorted: ✓ All pairs correct
hash_simple: ✓ All pairs correct
sort: ✓ All pairs correct

--------------- PERFORMANCE ANALYSIS ---------------
🏆 Fastest method: hash_sorted (0.0879s)

Speedup vs sort baseline (95.2078s):
  hash_cpu: 8.34x faster
  hash_sorted: 1083.35x faster
  hash_simple: 110.19x faster

--------------- LARGE MATRIX COMPARISON ---------------
✅ CPU Hash vs CUDA Hash Sorted: Results match perfectly (100.0%)
✅ CPU Hash vs CUDA Hash Simple: Results match perfectly (100.0%)
✅ CUDA Hash Sorted vs CUDA Hash Simple: Results match perfectly (100.0%)

================✅ BENCHMARK SUITE COMPLETED=================
Summary: All methods tested and compared successfully!
```

To build:
```
pip install --no-build-isolation -e .
```

To test:
```
python test/test_hash_pairing.py
```

Benchmarking:
```
python example_hash_pairing.py
```

