import torch
import unittest
import relu_pairing
from torch.testing._internal.common_utils import TestCase


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


class TestHashPairRows(TestCase):

    def test_simple_pairing_cpu(self):
        """Test with a simple 6x3 matrix on CPU"""
        # Create a simple test matrix with known pairs
        matrix = torch.tensor(
            [
                [2, 1, 0],  # row 0
                [0, 1, 2],  # row 1
                [1, 1, 1],  # row 2
                [1, 1, 1],  # row 3 (same as row 2)
                [2, 1, 0],  # row 4 (same as row 0)
                [0, 1, 2],  # row 5 (same as row 1)
            ],
            dtype=torch.long,
        )

        pairs = relu_pairing.ops.hash_pair_rows(matrix)

        # Should return 3 pairs
        self.assertEqual(pairs.shape, (3, 2))

        # Verify all pairs are valid
        for i in range(3):
            row1_idx = pairs[i, 0].item()
            row2_idx = pairs[i, 1].item()

            # Check that the paired rows are actually identical
            row1 = matrix[row1_idx]
            row2 = matrix[row2_idx]
            self.assertTrue(torch.equal(row1, row2))

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_simple_pairing_cuda(self):
        """Test with a simple 6x3 matrix on CUDA"""
        matrix = torch.tensor(
            [
                [2, 1, 0],
                [0, 1, 2],
                [1, 1, 1],
                [1, 1, 1],
                [2, 1, 0],
                [0, 1, 2],
            ],
            dtype=torch.long,
            device="cuda",
        )

        pairs = relu_pairing.ops.hash_pair_rows(matrix)

        # Should return 3 pairs
        self.assertEqual(pairs.shape, (3, 2))

        # Verify all pairs are valid
        for i in range(3):
            row1_idx = pairs[i, 0].item()
            row2_idx = pairs[i, 1].item()

            row1 = matrix[row1_idx]
            row2 = matrix[row2_idx]
            self.assertTrue(torch.equal(row1, row2))

    def test_large_matrix_cpu(self):
        """Test with larger random matrix on CPU"""
        # Create a 100x10 matrix with guaranteed pairs
        M, N = 100, 10
        base_rows = torch.randint(0, 3, (M // 2, N), dtype=torch.long)

        # Create pairs by duplicating each row
        matrix = torch.zeros(M, N, dtype=torch.long)
        for i in range(M // 2):
            matrix[2 * i] = base_rows[i]
            matrix[2 * i + 1] = base_rows[i]

        # Shuffle the rows to make it more challenging
        perm = torch.randperm(M)
        matrix = matrix[perm]

        pairs = relu_pairing.ops.hash_pair_rows(matrix)

        # Should return M/2 pairs
        self.assertEqual(pairs.shape, (M // 2, 2))

        # Verify all pairs are valid
        for i in range(M // 2):
            row1_idx = pairs[i, 0].item()
            row2_idx = pairs[i, 1].item()

            row1 = matrix[row1_idx]
            row2 = matrix[row2_idx]
            self.assertTrue(torch.equal(row1, row2))

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_large_matrix_cuda(self):
        """Test with larger random matrix on CUDA"""
        M, N = 100, 10
        base_rows = torch.randint(0, 3, (M // 2, N), dtype=torch.long)

        matrix = torch.zeros(M, N, dtype=torch.long)
        for i in range(M // 2):
            matrix[2 * i] = base_rows[i]
            matrix[2 * i + 1] = base_rows[i]

        perm = torch.randperm(M)
        matrix = matrix[perm].cuda()

        pairs = relu_pairing.ops.hash_pair_rows(matrix)

        self.assertEqual(pairs.shape, (M // 2, 2))

        # Verify all pairs are valid
        for i in range(M // 2):
            row1_idx = pairs[i, 0].item()
            row2_idx = pairs[i, 1].item()

            row1 = matrix[row1_idx]
            row2 = matrix[row2_idx]
            self.assertTrue(torch.equal(row1, row2))

    def test_data_file_matrix_6x3(self):
        """Test with the actual 6x3 data file"""
        try:
            matrix = load_matrix_from_file("./data/matrix_6x3.txt")
            pairs = relu_pairing.ops.hash_pair_rows(matrix)

            self.assertEqual(pairs.shape, (3, 2))

            # Verify pairs are actually identical
            for i in range(3):
                row1_idx = pairs[i, 0].item()
                row2_idx = pairs[i, 1].item()

                row1 = matrix[row1_idx]
                row2 = matrix[row2_idx]
                self.assertTrue(torch.equal(row1, row2))

            print(f"Found pairs in 6x3 matrix: {pairs}")
        except FileNotFoundError:
            print("Skipping data file test - file not found")

    def test_error_cases(self):
        """Test error handling"""
        # Test odd number of rows
        matrix_odd = torch.randint(0, 3, (5, 3), dtype=torch.long)
        with self.assertRaises(RuntimeError):
            relu_pairing.ops.hash_pair_rows(matrix_odd)

        # Test wrong dtype
        matrix_float = torch.randn(6, 3)
        with self.assertRaises(RuntimeError):
            relu_pairing.ops.hash_pair_rows(matrix_float)

        # Test 1D tensor
        matrix_1d = torch.randint(0, 3, (10,), dtype=torch.long)
        with self.assertRaises(RuntimeError):
            relu_pairing.ops.hash_pair_rows(matrix_1d)

    def test_cpu_cuda_consistency(self):
        """Test that CPU and CUDA implementations give same results"""
        if not torch.cuda.is_available():
            return

        # Create test matrix
        matrix_cpu = torch.tensor(
            [
                [1, 2, 3],
                [4, 5, 6],
                [1, 2, 3],  # duplicate of row 0
                [7, 8, 9],
                [4, 5, 6],  # duplicate of row 1
                [7, 8, 9],  # duplicate of row 3
            ],
            dtype=torch.long,
        )

        matrix_cuda = matrix_cpu.cuda()

        pairs_cpu = relu_pairing.ops.hash_pair_rows(matrix_cpu)
        pairs_cuda = relu_pairing.ops.hash_pair_rows(matrix_cuda)

        # Results should have same shape
        self.assertEqual(pairs_cpu.shape, pairs_cuda.shape)

        # Verify all pairs in both results are valid
        for i in range(pairs_cpu.shape[0]):
            cpu_row1_idx = pairs_cpu[i, 0].item()
            cpu_row2_idx = pairs_cpu[i, 1].item()
            cpu_row1 = matrix_cpu[cpu_row1_idx]
            cpu_row2 = matrix_cpu[cpu_row2_idx]
            self.assertTrue(torch.equal(cpu_row1, cpu_row2))

        for i in range(pairs_cuda.shape[0]):
            cuda_row1_idx = pairs_cuda[i, 0].item()
            cuda_row2_idx = pairs_cuda[i, 1].item()
            cuda_row1 = matrix_cuda[cuda_row1_idx]
            cuda_row2 = matrix_cuda[cuda_row2_idx]
            self.assertTrue(torch.equal(cuda_row1, cuda_row2))


if __name__ == "__main__":
    unittest.main()
