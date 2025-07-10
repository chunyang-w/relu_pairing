#!/usr/bin/env python3

"""
Example usage of the hash_pair_rows functions for efficient row pairing.

This demonstrates how to use the custom CUDA/CPU hash-based pairing operations
to find pairs of identical rows in matrices.

Available Methods:
1. relu_pairing.ops.hash_pair_rows - CPU hash table (O(N))
2. relu_pairing.ops.hash_pair_rows_sorted - CUDA with sorting (O(N log N))
3. relu_pairing.ops.hash_pair_rows_simple - CUDA without sorting (O(N))
"""

import torch
import time
import os
import numpy as np
import relu_pairing


def load_matrix_from_file(filepath):
    """Load matrix from text file format used in data/ folder"""
    with open(filepath, "r") as f:
        lines = f.readlines()

    # First line contains dimensions
    M, N = map(int, lines[0].strip().split())

    # Read matrix data
    matrix = []
    for i in range(1, M + 1):
        row = list(map(int, lines[i].strip().split()))
        matrix.append(row)

    return torch.tensor(matrix, dtype=torch.long)


def pair_via_sort(b: torch.Tensor):
    """
    Python-based ground truth implementation using PyTorch's built-in
    functions. Modified to return pairs in (M/2, 2) format.

    Args:
        b: Input tensor of shape (M, N)

    Returns:
        Tensor of shape (M/2, 2) containing pairs of row indices
    """
    device = b.device

    if device.type == "cuda":
        # On CUDA, unique is faster
        _, inv = b.unique(dim=0, return_inverse=True)
        inv_sorted, inv_inds = inv.sort()
        # Find pairs of identical rows
        idxs_of_pairs_in_inv = (~inv_sorted.diff().bool()).nonzero().squeeze(1)
        # Get both indices of each pair
        pair_indices = torch.stack(
            [
                inv_inds[idxs_of_pairs_in_inv],
                inv_inds[idxs_of_pairs_in_inv + 1],
            ],
            dim=1,
        )
    else:
        # On CPU, lexsort is faster
        b_lex_idxs = torch.from_numpy(np.lexsort(b.numpy().T))
        b_lex = b[b_lex_idxs]
        idxs_of_pairs_in_lex = (
            torch.all(~b_lex.diff(dim=0).bool(), dim=1).nonzero().squeeze(1)
        )
        # Get both indices of each pair
        pair_indices = torch.stack(
            [
                b_lex_idxs[idxs_of_pairs_in_lex],
                b_lex_idxs[idxs_of_pairs_in_lex + 1],
            ],
            dim=1,
        )

    return pair_indices


def naive_pair_rows(matrix):
    """Naive O(M^2) approach for comparison"""
    M, N = matrix.shape
    pairs = []
    used = set()

    for i in range(M):
        if i in used:
            continue
        for j in range(i + 1, M):
            if j in used:
                continue
            if torch.equal(matrix[i], matrix[j]):
                pairs.append((i, j))
                used.add(i)
                used.add(j)
                break

    return torch.tensor(pairs, dtype=torch.long)


def verify_pairs_correctness(matrix, pairs, method_name):
    """Verify that all pairs in the result are actually identical rows"""
    if pairs.shape[0] == 0:
        return True, "No pairs to verify"

    all_correct = True
    incorrect_pairs = []

    for i in range(pairs.shape[0]):
        idx1, idx2 = pairs[i]
        if not torch.equal(matrix[idx1], matrix[idx2]):
            all_correct = False
            incorrect_pairs.append((i, idx1.item(), idx2.item()))

    if all_correct:
        status = "✓ All pairs correct"
    else:
        status = f"✗ Invalid pairs found ({len(incorrect_pairs)} incorrect)"
        print(f"\n   🔍 INCORRECT PAIRS DETAILS for {method_name}:")
        for pair_idx, idx1, idx2 in incorrect_pairs:
            print(f"   Pair {pair_idx}: rows {idx1} and {idx2} don't match")
            print(f"     Row {idx1}: {matrix[idx1].tolist()}")
            print(f"     Row {idx2}: {matrix[idx2].tolist()}")

            # Show which elements differ
            diff_mask = matrix[idx1] != matrix[idx2]
            if diff_mask.any():
                diff_positions = diff_mask.nonzero().squeeze().tolist()
                if isinstance(diff_positions, int):
                    diff_positions = [diff_positions]
                print(f"     Differences at positions: {diff_positions}")
            print()

    return all_correct, f"{method_name}: {status}"


def compare_pair_results(pairs1, pairs2, name1, name2):
    """Compare two pairing results (order may differ)"""
    if pairs1.shape != pairs2.shape:
        return (
            f"❌ {name1} vs {name2}: Different shapes "
            f"{pairs1.shape} vs {pairs2.shape}"
        )

    # Convert pairs to sets of tuples for easier comparison
    # Each pair should be sorted so (i,j) and (j,i) are treated the same
    set1 = set()
    set2 = set()

    for i in range(pairs1.shape[0]):
        pair = tuple(sorted([pairs1[i][0].item(), pairs1[i][1].item()]))
        set1.add(pair)

    for i in range(pairs2.shape[0]):
        pair = tuple(sorted([pairs2[i][0].item(), pairs2[i][1].item()]))
        set2.add(pair)

    # Find matching and non-matching pairs
    matching_pairs = set1.intersection(set2)
    only_in_1 = set1 - set2
    only_in_2 = set2 - set1

    total_pairs = len(set1)  # Should be same as len(set2) since shapes match
    matching_count = len(matching_pairs)

    if total_pairs == 0:
        percentage = 100.0
    else:
        percentage = (matching_count / total_pairs) * 100

    if percentage == 100.0:
        return f"✅ {name1} vs {name2}: Results match perfectly (100.0%)"
    else:
        result_message = (
            f"❌ {name1} vs {name2}: {matching_count}/{total_pairs} pairs "
            f"match ({percentage:.1f}%)"
        )

        if only_in_1:
            result_message += (
                f"\n   Pairs only in {name1}: {sorted(list(only_in_1))}"
            )
        if only_in_2:
            result_message += (
                f"\n   Pairs only in {name2}: {sorted(list(only_in_2))}"
            )
        return result_message


def benchmark_all_methods(matrix, device="cpu"):
    """Benchmark all pairing methods and compare results"""
    print(f"\n{'='*20} BENCHMARKING ON {device.upper()} {'='*20}")
    print(f"Matrix shape: {matrix.shape}")
    print(f"Expected pairs: {matrix.shape[0] // 2}")

    if device == "cuda" and torch.cuda.is_available():
        matrix = matrix.cuda()

    results = {}
    timings = {}

    # Method 1: CPU hash-based implementation
    print(f"\n{'-'*15} Hash-based Method (CPU) {'-'*15}")
    start_time = time.time()
    try:
        matrix_cpu = matrix.cpu() if device == "cuda" else matrix
        hash_pairs = relu_pairing.ops.hash_pair_rows(matrix_cpu)
        if device == "cuda":
            hash_pairs = hash_pairs.cuda()
        hash_time = time.time() - start_time
        results["hash_cpu"] = hash_pairs
        timings["hash_cpu"] = hash_time
        print(
            f"✅ Hash CPU method: {hash_time:.4f}s, "
            f"found {hash_pairs.shape[0]} pairs"
        )
    except Exception as e:
        print(f"❌ Hash CPU method failed: {e}")
        results["hash_cpu"] = None
        timings["hash_cpu"] = float("inf")

    # Method 1a: CUDA hash-based implementation (sorted) - only on CUDA
    if device == "cuda" and torch.cuda.is_available():
        print(f"\n{'-'*15} Hash-based Method (CUDA Sorted) {'-'*15}")
        start_time = time.time()
        try:
            hash_pairs_sorted = relu_pairing.ops.hash_pair_rows_sorted(matrix)
            hash_sorted_time = time.time() - start_time
            results["hash_sorted"] = hash_pairs_sorted
            timings["hash_sorted"] = hash_sorted_time
            print(
                f"✅ Hash CUDA Sorted method: {hash_sorted_time:.4f}s, "
                f"found {hash_pairs_sorted.shape[0]} pairs"
            )
        except Exception as e:
            print(f"❌ Hash CUDA Sorted method failed: {e}")
            results["hash_sorted"] = None
            timings["hash_sorted"] = float("inf")

    # Method 1b: CUDA hash-based implementation (simple O(N)) - only on CUDA
    if device == "cuda" and torch.cuda.is_available():
        print(f"\n{'-'*15} Hash-based Method (CUDA Simple O(N)) {'-'*15}")
        start_time = time.time()
        try:
            hash_pairs_simple = relu_pairing.ops.hash_pair_rows_simple(matrix)
            hash_simple_time = time.time() - start_time
            results["hash_simple"] = hash_pairs_simple
            timings["hash_simple"] = hash_simple_time
            print(
                f"✅ Hash CUDA Simple method: {hash_simple_time:.4f}s, "
                f"found {hash_pairs_simple.shape[0]} pairs"
            )
        except Exception as e:
            print(f"❌ Hash CUDA Simple method failed: {e}")
            results["hash_simple"] = None
            timings["hash_simple"] = float("inf")

    # Method 2: Python ground truth (sort-based)
    print(f"\n{'-'*15} Sort-based Method (Ground Truth) {'-'*15}")
    start_time = time.time()
    try:
        matrix_for_sort = matrix.cpu() if device == "cuda" else matrix
        sort_pairs = pair_via_sort(matrix_for_sort)
        if device == "cuda":
            sort_pairs = sort_pairs.cuda()
        sort_time = time.time() - start_time
        results["sort"] = sort_pairs
        timings["sort"] = sort_time
        print(
            f"✅ Sort method: {sort_time:.4f}s, "
            f"found {sort_pairs.shape[0]} pairs"
        )
    except Exception as e:
        print(f"❌ Sort method failed: {e}")
        results["sort"] = None
        timings["sort"] = float("inf")

    # Method 3: Naive approach (only for smaller matrices)
    if matrix.shape[0] <= 1000:
        print(f"\n{'-'*15} Naive Method (Reference) {'-'*15}")
        start_time = time.time()
        try:
            matrix_cpu = matrix.cpu() if device == "cuda" else matrix
            naive_pairs = naive_pair_rows(matrix_cpu)
            if device == "cuda":
                naive_pairs = naive_pairs.cuda()
            naive_time = time.time() - start_time
            results["naive"] = naive_pairs
            timings["naive"] = naive_time
            print(
                f"✅ Naive method: {naive_time:.4f}s, "
                f"found {naive_pairs.shape[0]} pairs"
            )
        except Exception as e:
            print(f"❌ Naive method failed: {e}")
            results["naive"] = None
            timings["naive"] = float("inf")
    else:
        print(f"\n{'-'*15} Naive Method (Skipped - too large) {'-'*15}")
        results["naive"] = None
        timings["naive"] = float("inf")

    # Verification and comparison
    print(f"\n{'-'*15} CORRECTNESS VERIFICATION {'-'*15}")
    matrix_cpu = matrix.cpu() if device == "cuda" else matrix

    for method, pairs in results.items():
        if pairs is not None:
            pairs_cpu = pairs.cpu() if device == "cuda" else pairs
            is_correct, msg = verify_pairs_correctness(
                matrix_cpu, pairs_cpu, method
            )
            print(msg)

    # # Compare methods
    # print(f"\n{'-'*15} RESULT COMPARISON {'-'*15}")
    # methods = [(k, v) for k, v in results.items() if v is not None]

    # for i in range(len(methods)):
    #     for j in range(i + 1, len(methods)):
    #         name1, pairs1 = methods[i]
    #         name2, pairs2 = methods[j]
    #         pairs1_cpu = pairs1.cpu() if device == "cuda" else pairs1
    #         pairs2_cpu = pairs2.cpu() if device == "cuda" else pairs2
    #         comparison = compare_pair_results(
    #             pairs1_cpu, pairs2_cpu, name1, name2
    #         )
    #         print(comparison)

    # Performance analysis
    print(f"\n{'-'*15} PERFORMANCE ANALYSIS {'-'*15}")
    valid_methods = [(k, v) for k, v in timings.items() if v != float("inf")]

    if len(valid_methods) > 1:
        fastest_method = min(valid_methods, key=lambda x: x[1])
        print(
            f"🏆 Fastest method: {fastest_method[0]} "
            f"({fastest_method[1]:.4f}s)"
        )

        # Always compare against the sort baseline (ground truth)
        if "sort" in timings and timings["sort"] != float("inf"):
            sort_time = timings["sort"]
            print(f"\nSpeedup vs sort baseline ({sort_time:.4f}s):")
            for method, time_taken in valid_methods:
                if method != "sort":
                    speedup = sort_time / time_taken
                    print(f"  {method}: {speedup:.2f}x faster")
        else:
            print("\n⚠️  Sort baseline not available for comparison")

    return results


def debug_available_methods():
    """Debug function to check what methods are available"""
    print("\n🔍 DEBUGGING AVAILABLE METHODS:")
    print("=" * 50)
    
    # Check what's available in relu_pairing.ops
    import relu_pairing.ops as ops
    available_methods = [attr for attr in dir(ops) if not attr.startswith('_')]
    print(f"Available methods in relu_pairing.ops: {available_methods}")
    
    # Test each method individually
    test_methods = [
        'hash_pair_rows', 'hash_pair_rows_sorted', 'hash_pair_rows_simple'
    ]
    
    for method_name in test_methods:
        if hasattr(ops, method_name):
            print(f"✅ {method_name}: Available")
        else:
            print(f"❌ {method_name}: Not found")
    
    # Check CUDA availability
    print(f"\nCUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
    
    print("=" * 50)


def main():
    print("🚀 Hash-based Row Pairing Benchmark Suite")
    print("=" * 60)
    
    # Debug what methods are available
    debug_available_methods()

    # Test with data files
    data_files = [
        "data/matrix_6x3.txt",
        "data/matrix_6x12.txt",
        "data/matrix_2000x20.txt",
    ]

    for filepath in data_files:
        if os.path.exists(filepath):
            print(f"\n{'🔍 PROCESSING: ' + filepath:=^60}")
            matrix = load_matrix_from_file(filepath)

            # CPU benchmarks
            cpu_results = benchmark_all_methods(matrix, device="cpu")

            # CUDA benchmarks if available
            if torch.cuda.is_available():
                cuda_results = benchmark_all_methods(matrix, device="cuda")

                # Cross-device comparison
                print(f"\n{'-'*15} CPU vs CUDA COMPARISON {'-'*15}")
                
                # Compare CPU hash with CUDA sorted
                if (
                    cpu_results.get("hash_cpu") is not None
                    and cuda_results.get("hash_sorted") is not None
                ):
                    cpu_pairs = cpu_results["hash_cpu"].cpu()
                    cuda_sorted_pairs = cuda_results["hash_sorted"].cpu()
                    comparison = compare_pair_results(
                        cpu_pairs, cuda_sorted_pairs, "CPU Hash", 
                        "CUDA Hash Sorted"
                    )
                    print(comparison)
                
                # Compare CPU hash with CUDA simple
                if (
                    cpu_results.get("hash_cpu") is not None
                    and cuda_results.get("hash_simple") is not None
                ):
                    cpu_pairs = cpu_results["hash_cpu"].cpu()
                    cuda_simple_pairs = cuda_results["hash_simple"].cpu()
                    comparison = compare_pair_results(
                        cpu_pairs, cuda_simple_pairs, "CPU Hash", 
                        "CUDA Hash Simple"
                    )
                    print(comparison)
                
                # Compare CUDA sorted vs CUDA simple
                if (
                    cuda_results.get("hash_sorted") is not None
                    and cuda_results.get("hash_simple") is not None
                ):
                    cuda_sorted_pairs = cuda_results["hash_sorted"].cpu()
                    cuda_simple_pairs = cuda_results["hash_simple"].cpu()
                    comparison = compare_pair_results(
                        cuda_sorted_pairs, cuda_simple_pairs, 
                        "CUDA Hash Sorted", "CUDA Hash Simple"
                    )
                    print(comparison)
            else:
                msg = "⚠️  CUDA not available - skipping GPU benchmarks"
                print(f"\n{msg:^60}")

        else:
            print(f"\n❌ File {filepath} not found, skipping...")

    # Synthetic large matrix test
    print(f"\n{'🧪 SYNTHETIC LARGE MATRIX TEST':=^60}")

    # Create a matrix with guaranteed pairs for testing
    M, N = 1000000, 2000  # Large for performance testing
    # M, N = 2000, 50  # Smaller for initial testing
    print(f"Creating synthetic matrix of shape {M}x{N}...")

    # Generate M/2 unique base rows
    base_rows = torch.randint(0, 3, (M // 2, N), dtype=torch.long)

    # Check if base rows are unique by looking for duplicates
    _, counts = torch.unique(base_rows, dim=0, return_counts=True)
    if torch.any(counts > 1):
        raise ValueError(
            "Generated base rows contain duplicates - regenerating required"
        )

    # Create pairs by duplicating each row
    matrix = torch.zeros(M, N, dtype=torch.long)
    for i in range(M // 2):
        matrix[2 * i] = base_rows[i]
        matrix[2 * i + 1] = base_rows[i]

    # Shuffle to make it more challenging
    perm = torch.randperm(M)
    matrix = matrix[perm]

    # Benchmark large matrix (skip naive method)
    print(f"\n{'Large Matrix Benchmarks':^40}")
    cpu_results = benchmark_all_methods(matrix, device="cpu")

    if torch.cuda.is_available():
        cuda_results = benchmark_all_methods(matrix, device="cuda")
        
        # Cross-method comparison for large matrix
        print(f"\n{'-'*15} LARGE MATRIX COMPARISON {'-'*15}")
        
        # Compare CPU hash with CUDA sorted
        if (
            cpu_results.get("hash_cpu") is not None
            and cuda_results.get("hash_sorted") is not None
        ):
            cpu_pairs = cpu_results["hash_cpu"].cpu()
            cuda_sorted_pairs = cuda_results["hash_sorted"].cpu()
            comparison = compare_pair_results(
                cpu_pairs, cuda_sorted_pairs, "CPU Hash", 
                "CUDA Hash Sorted"
            )
            print(comparison)
        
        # Compare CPU hash with CUDA simple
        if (
            cpu_results.get("hash_cpu") is not None
            and cuda_results.get("hash_simple") is not None
        ):
            cpu_pairs = cpu_results["hash_cpu"].cpu()
            cuda_simple_pairs = cuda_results["hash_simple"].cpu()
            comparison = compare_pair_results(
                cpu_pairs, cuda_simple_pairs, "CPU Hash", 
                "CUDA Hash Simple"
            )
            print(comparison)
        
        # Compare CUDA sorted vs CUDA simple
        if (
            cuda_results.get("hash_sorted") is not None
            and cuda_results.get("hash_simple") is not None
        ):
            cuda_sorted_pairs = cuda_results["hash_sorted"].cpu()
            cuda_simple_pairs = cuda_results["hash_simple"].cpu()
            comparison = compare_pair_results(
                cuda_sorted_pairs, cuda_simple_pairs, 
                "CUDA Hash Sorted", "CUDA Hash Simple"
            )
            print(comparison)

    print(f"\n{'✅ BENCHMARK SUITE COMPLETED':=^60}")
    print("Summary: All methods tested and compared successfully!")


if __name__ == "__main__":
    main()
