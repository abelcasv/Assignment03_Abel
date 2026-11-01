import csv
import hashlib
import os
import random
import sys
import time

DENSE_SIZES = [250, 500, 750, 1000, 1500, 2000]
VALIDATION_SIZE = 4
SEED = 42


def write_csv_matrix(filepath: str, matrix: list[list[float]]) -> None:
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for row in matrix:
            writer.writerow([f"{val:.6f}" for val in row])


def reference_multiply(matrix_a: list[list[float]],
                       matrix_b: list[list[float]],
                       n: int) -> list[list[float]]:
    c = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            s = 0.0
            for k in range(n):
                s += matrix_a[i][k] * matrix_b[k][j]
            c[i][j] = s
    return c


def generate_validation_matrices(data_dir: str) -> None:
    val_dir = os.path.join(data_dir, "validation")
    os.makedirs(val_dir, exist_ok=True)
    n = VALIDATION_SIZE

    identity = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    zero = [[0.0] * n for _ in range(n)]

    known_a = [
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0, 7.0],
    ]
    known_b = [
        [7.0, 6.0, 5.0, 4.0],
        [3.0, 2.0, 1.0, 9.0],
        [8.0, 7.0, 6.0, 5.0],
        [1.0, 2.0, 3.0, 4.0],
    ]
    expected_c = reference_multiply(known_a, known_b, n)

    write_csv_matrix(os.path.join(val_dir, "identity_4.csv"), identity)
    write_csv_matrix(os.path.join(val_dir, "zero_4.csv"), zero)
    write_csv_matrix(os.path.join(val_dir, "known_A_4.csv"), known_a)
    write_csv_matrix(os.path.join(val_dir, "known_B_4.csv"), known_b)
    write_csv_matrix(os.path.join(val_dir, "expected_C_4.csv"), expected_c)


def generate_dense_matrices(data_dir: str, rng: random.Random) -> None:
    dense_dir = os.path.join(data_dir, "dense")
    os.makedirs(dense_dir, exist_ok=True)

    for n in DENSE_SIZES:
        file_a = os.path.join(dense_dir, f"A_{n}.csv")
        file_b = os.path.join(dense_dir, f"B_{n}.csv")

        matrix_a = [[rng.random() for _ in range(n)] for _ in range(n)]
        matrix_b = [[rng.random() for _ in range(n)] for _ in range(n)]

        write_csv_matrix(file_a, matrix_a)
        write_csv_matrix(file_b, matrix_b)


def write_sha256_manifest(data_dir: str) -> None:
    data_root = os.path.abspath(data_dir)
    matrix_files = []
    for subdirectory in ("validation", "dense"):
        folder = os.path.join(data_root, subdirectory)
        matrix_files.extend(
            os.path.join(folder, name)
            for name in os.listdir(folder)
            if name.endswith(".csv")
        )

    lines = ["# SHA-256 checksums for generated validation fixtures and dense inputs."]
    for filepath in sorted(matrix_files):
        digest = hashlib.sha256()
        with open(filepath, "rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        relative_path = os.path.relpath(filepath, data_root).replace(os.sep, "/")
        lines.append(f"{digest.hexdigest()}  {relative_path}")

    manifest_path = os.path.join(data_root, "SHA256SUMS.txt")
    with open(manifest_path, "w", encoding="utf-8", newline="\n") as manifest:
        manifest.write("\n".join(lines) + "\n")


def main() -> None:
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    generate_validation_matrices(data_dir)
    rng = random.Random(SEED)
    generate_dense_matrices(data_dir, rng)
    write_sha256_manifest(data_dir)


if __name__ == "__main__":
    main()
