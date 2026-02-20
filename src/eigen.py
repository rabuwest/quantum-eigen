import numpy as np
import argparse
import time
from scipy.linalg import eigh


def build_2d_hamiltonian(N=20, potential='well'):
    
    """
    Build a discretized 2D Hamiltonian on an N x N grid.
    """

    dx = 1.0 / float(N)
    inv_dx2 = float(N * N)  # 1/dx^2

    H = np.zeros((N * N, N * N), dtype=np.float64)

    # Map (i, j) -> linear index
    def idx(i, j):
        return i * N + j

    # Potential function
    def V(i, j):
        if potential == 'well':
            return 0.0
        elif potential == 'harmonic':
            x = (i - N / 2) * dx
            y = (j - N / 2) * dx
            return 4.0 * (x**2 + y**2)
        else:
            raise ValueError(f"Unknown potential type '{potential}'")

    # Build Hamiltonian matrix
    for i in range(N):
        for j in range(N):
            row = idx(i, j)

            # Diagonal term (2D Laplacian + V)
            H[row, row] = -4.0 * inv_dx2 + V(i, j)

            # Neighbors (finite difference Laplacian)
            if i > 0:
                H[row, idx(i - 1, j)] = inv_dx2
            if i < N - 1:
                H[row, idx(i + 1, j)] = inv_dx2
            if j > 0:
                H[row, idx(i, j - 1)] = inv_dx2
            if j < N - 1:
                H[row, idx(i, j + 1)] = inv_dx2

    return H


def solve_eigen(N=20, potential='well', n_eigs=None):
    """
    Solve for eigenvalues of 2D Hamiltonian.
    """

    # --- Sanity checks ---

    if not isinstance(N, int) or N <= 0:
        raise ValueError("N must be a positive integer.")

    valid_potentials = ['well', 'harmonic']
    if potential not in valid_potentials:
        raise ValueError(f"Potential must be one of {valid_potentials}")

    max_dim = N * N
    if n_eigs is not None:
        if not isinstance(n_eigs, int) or n_eigs <= 0:
            raise ValueError("n_eigs must be a positive integer.")
        if n_eigs > max_dim:
            raise ValueError(f"n_eigs cannot exceed N^2 = {max_dim}")
    
    # --- End of Sanity Checks ---


    # --- Build and solve ---
    H = build_2d_hamiltonian(N, potential)

    vals, vecs = eigh(H)

    idx_sorted = np.argsort(vals)
    vals_sorted = vals[idx_sorted]
    vecs_sorted = vecs[:, idx_sorted]

    if n_eigs is None:
        return vals_sorted, vecs_sorted
    else:
        return vals_sorted[:n_eigs], vecs_sorted[:, :n_eigs]


def main():
    parser = argparse.ArgumentParser(
        description="Solve a discretized 2D Schrödinger Hamiltonian."
    )

    parser.add_argument(
        "--N",
        type=int,
        default=20,
        help="Grid size in each dimension (default: 20)"
    )

    parser.add_argument(
        "--potential",
        type=str,
        choices=["well", "harmonic"],
        default="well",
        help="Type of potential (well or harmonic)"
    )

    parser.add_argument(
        "--neigs",
        type=int,
        default=None,
        help="Number of lowest eigenvalues to compute (default: all)"
    )

    args = parser.parse_args()

    try:
        vals, vecs = solve_eigen(
            N=args.N,
            potential=args.potential,
            n_eigs=args.neigs
        )

        if args.neigs is None:
            print("Computed full spectrum.")
        else:
            print(f"Lowest {args.neigs} eigenvalues:")

        #print(vals)

    except Exception as e:
        print("Error:", e)

    start = time.time()
    vals = solve_eigen(args.N, args.potential, args.neigs)[0]
    end = time.time()

    runtime = end - start

    vals = np.ravel(vals)
                     
    # Save eigenvalues
    np.savetxt(f"eigs_N{args.N}.txt", vals)

    # Append runtime info
    with open("timing_results.txt", "a") as f:
        f.write(f"N={args.N}, runtime={runtime:.6f} sec\n")

    print(f"N={args.N} finished in {runtime:.4f} seconds")
if __name__ == "__main__":
    main()
