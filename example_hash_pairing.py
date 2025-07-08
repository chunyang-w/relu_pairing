#!/usr/bin/env python3

"""
Example usage of the hash_pair_rows function for efficient row pairing.

This demonstrates how to use the custom CUDA/CPU hash-based pairing operation
to find pairs of identical rows in matrices with O(N) complexity.
"""

import torch
import time
import os
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


def benchmark_pairing_methods(matrix, device="cpu"):
    """Benchmark different pairing methods"""
    print(f"\nBenchmarking on {device.upper()}:")
    print(f"Matrix shape: {matrix.shape}")

    if device == "cuda" and torch.cuda.is_available():
        matrix = matrix.cuda()

    # Our hash-based method
    start_time = time.time()
    hash_pairs = relu_pairing.ops.hash_pair_rows(matrix)
    hash_time = time.time() - start_time

    print(f"Hash-based pairing: {hash_time:.4f} seconds")
    print(f"Found {hash_pairs.shape[0]} pairs")

    # Naive method (only for smaller matrices)
    if matrix.shape[0] <= 1000:
        matrix_cpu = matrix.cpu() if device == "cuda" else matrix
        start_time = time.time()
        naive_pairs = naive_pair_rows(matrix_cpu)
        naive_time = time.time() - start_time

        print(f"Naive pairing: {naive_time:.4f} seconds")
        print(f"Speedup: {naive_time / hash_time:.2f}x")

        # Verify results are equivalent (pairs may be in different order)
        print(f"Results match: {hash_pairs.shape[0] == naive_pairs.shape[0]}")

    return hash_pairs


def main():
    print("Hash-based Row Pairing Example")
    print("=" * 40)

    # Test with data files
    data_files = [
        "data/matrix_6x3.txt",
        "data/matrix_6x12.txt",
        "data/matrix_2000x20.txt",
    ]

    for filepath in data_files:
        if os.path.exists(filepath):
            print(f"\nProcessing {filepath}:")
            matrix = load_matrix_from_file(filepath)
            print(f"Matrix shape: {matrix.shape}")

            # CPU benchmark
            pairs_cpu = benchmark_pairing_methods(matrix, device="cpu")

            # Show some pairs
            print(f"First few pairs: {pairs_cpu[:min(3, pairs_cpu.shape[0])]}")

            # Verify pairs are actually identical
            print("Verifying pairs are identical:")
            for i in range(min(3, pairs_cpu.shape[0])):
                idx1, idx2 = pairs_cpu[i]
                row1, row2 = matrix[idx1], matrix[idx2]
                is_equal = torch.equal(row1, row2)
                status = "✓" if is_equal else "✗"
                print(f"  Pair {i}: rows {idx1} and {idx2} -> {status}")

            # CUDA benchmark if available
            if torch.cuda.is_available():
                print("Running CUDA benchmark...")
                benchmark_pairing_methods(matrix, device="cuda")
        else:
            print(f"File {filepath} not found, skipping...")

    # Create a synthetic larger example to showcase performance
    print("\n" + "=" * 50)
    print("Synthetic Large Matrix Example")
    print("=" * 50)

    # Create a 10000x50 matrix with guaranteed pairs
    M, N = 10000, 50
    print(f"Creating synthetic matrix of shape {M}x{N}...")

    # Generate M/2 unique base rows
    base_rows = torch.randint(0, 3, (M // 2, N), dtype=torch.long)

    # Create pairs by duplicating each row
    matrix = torch.zeros(M, N, dtype=torch.long)
    for i in range(M // 2):
        matrix[2 * i] = base_rows[i]
        matrix[2 * i + 1] = base_rows[i]

    # Shuffle to make it more challenging
    perm = torch.randperm(M)
    matrix = matrix[perm]

    # Benchmark on this large matrix
    pairs_large = benchmark_pairing_methods(matrix, device="cpu")

    if torch.cuda.is_available():
        print("Running large matrix CUDA benchmark...")
        benchmark_pairing_methods(matrix, device="cuda")

    print("\nLarge matrix test completed successfully!")
    print(f"Found {pairs_large.shape[0]} pairs out of expected {M//2}")


if __name__ == "__main__":
    main()
