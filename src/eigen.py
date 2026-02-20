import numpy as np
import argparse
import time
from scipy.linalg import eigh
import matplotlib.pyplot as plt


def build_2d_hamiltonian(N=20, potential='well', bc_type='none', bc_params=None):
    
    """
    Build a discretized 2D Hamiltonian on an N x N grid with optional boundary conditions.
    
    Parameters:
    -----------
    N : int
        Grid size in each dimension
    potential : str
        Type of potential ('well' or 'harmonic')
    bc_type : str
        Type of boundary condition: 'none', 'dirichlet_linear'
    bc_params : dict, optional
        Parameters for boundary conditions. For 'dirichlet_linear', use:
        {'a': coefficient_x, 'b': coefficient_y} for f(x,y) = ax + by
    """

    dx = 1.0 / float(N)
    inv_dx2 = float(N * N)  # 1/dx^2

    H = np.zeros((N * N, N * N), dtype=np.float64)

    # Map (i, j) -> linear index
    def idx(i, j):
        return i * N + j

    # Potential function
    def V(i, j):
        x = (i - N / 2) * dx
        y = (j - N / 2) * dx
        
        if potential == 'well':
            return 0.0
        elif potential == 'harmonic':
            return 4.0 * (x**2 + y**2)
        elif potential == 'double_well':
            # Double-well potential: V(x,y) = (x^2 - 1)^2 + 4*y^2
            return (x**2 - 0.5)**2 + 4.0 * y**2
        elif potential == 'anisotropic_harmonic':
            # Anisotropic harmonic oscillator: V(x,y) = 2*x^2 + 8*y^2
            # Different frequencies in x and y directions
            return 2.0 * x**2 + 8.0 * y**2
        elif potential == 'quartic':
            # Quartic potential: V(x,y) = (x^4 + y^4)
            return x**4 + y**4
        else:
            raise ValueError(f"Unknown potential type '{potential}'")

    # Boundary condition function
    def bc_value(i, j):
        """Return boundary condition value at grid point (i, j)"""
        if bc_type == 'none':
            return 0.0
        elif bc_type == 'dirichlet_linear':
            if bc_params is None:
                a, b = 0.0, 0.0
            else:
                a = bc_params.get('a', 0.0)
                b = bc_params.get('b', 0.0)
            # Map grid indices to physical coordinates
            x = (i - N / 2) * dx
            y = (j - N / 2) * dx
            return a * x + b * y
        else:
            raise ValueError(f"Unknown boundary condition type '{bc_type}'")

    # Check if a point is on the boundary
    def is_boundary(i, j):
        return i == 0 or i == N - 1 or j == 0 or j == N - 1

    # Build Hamiltonian matrix
    for i in range(N):
        for j in range(N):
            row = idx(i, j)

            if is_boundary(i, j) and bc_type != 'none':
                # Apply Dirichlet boundary condition
                H[row, row] = 1.0
                # BC value is handled separately in solving
            else:
                # Interior point: standard finite difference
                # Diagonal term (2D Laplacian + V)
                H[row, row] = -4.0 * inv_dx2 + V(i, j)

                # Neighbors (finite difference Laplacian)
                if i > 0:
                    if is_boundary(i - 1, j) and bc_type != 'none':
                        # Neighbor on boundary: contribution handled via RHS
                        pass
                    else:
                        H[row, idx(i - 1, j)] = inv_dx2
                if i < N - 1:
                    if is_boundary(i + 1, j) and bc_type != 'none':
                        pass
                    else:
                        H[row, idx(i + 1, j)] = inv_dx2
                if j > 0:
                    if is_boundary(i, j - 1) and bc_type != 'none':
                        pass
                    else:
                        H[row, idx(i, j - 1)] = inv_dx2
                if j < N - 1:
                    if is_boundary(i, j + 1) and bc_type != 'none':
                        pass
                    else:
                        H[row, idx(i, j + 1)] = inv_dx2

    return H


def solve_eigen(N=20, potential='well', n_eigs=None, bc_type='none', bc_params=None):
    """
    Solve for eigenvalues of 2D Hamiltonian with optional boundary conditions.
    
    Parameters:
    -----------
    N : int
        Grid size in each dimension
    potential : str
        Type of potential: 'well', 'harmonic', 'double_well', 'anisotropic_harmonic', 'quartic'
    n_eigs : int, optional
        Number of lowest eigenvalues to compute
    bc_type : str
        Type of boundary condition: 'none', 'dirichlet_linear'
    bc_params : dict, optional
        Parameters for boundary conditions
    """

    # --- Sanity checks ---

    if not isinstance(N, int) or N <= 0:
        raise ValueError("N must be a positive integer.")

    valid_potentials = ['well', 'harmonic', 'double_well', 'anisotropic_harmonic', 'quartic']
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
    H = build_2d_hamiltonian(N, potential, bc_type, bc_params)

    vals, vecs = eigh(H)

    idx_sorted = np.argsort(vals)
    vals_sorted = vals[idx_sorted]
    vecs_sorted = vecs[:, idx_sorted]

    if n_eigs is None:
        return vals_sorted, vecs_sorted
    else:
        return vals_sorted[:n_eigs], vecs_sorted[:, :n_eigs]


def visualize_ground_state(ground_state_vec, N=20, potential='well', save_file=None):
    """
    Visualize and optionally save the ground-state probability density |ψ(x, y)|^2.
    
    Parameters:
    -----------
    ground_state_vec : array
        Eigenvector of length N^2 corresponding to the ground state
    N : int
        Grid size in each dimension
    potential : str
        Type of potential ('well' or 'harmonic')
    save_file : str, optional
        If provided, save the probability density to this file
    """
    # Reshape the vector from (N^2,) to (N, N)
    psi = ground_state_vec.reshape((N, N))
    
    # Compute probability density |ψ|^2
    prob_density = np.abs(psi)**2
    
    # Save if requested
    if save_file:
        np.savetxt(save_file, prob_density)
        print(f"Probability density saved to {save_file}")
    
    # Plot the probability density
    dx = 1.0 / float(N)
    extent = (-(N/2)*dx, (N/2)*dx, -(N/2)*dx, (N/2)*dx)
    
    plt.figure(figsize=(8, 7))
    plt.imshow(prob_density.T, origin='lower', extent=extent, cmap='viridis')
    plt.colorbar(label='|ψ(x,y)|²')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Ground State Probability Density (N={N}, {potential} potential)')
    plt.show()



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
        choices=["well", "harmonic", "double_well", "anisotropic_harmonic", "quartic"],
        default="well",
        help="Type of potential (well, harmonic, double_well, anisotropic_harmonic, or quartic)"
    )

    parser.add_argument(
        "--neigs",
        type=int,
        default=None,
        help="Number of lowest eigenvalues to compute (default: all)"
    )

    parser.add_argument(
        "--save-ground-state",
        action="store_true",
        help="Save the ground-state probability density to a file"
    )

    parser.add_argument(
        "--bc-type",
        type=str,
        choices=["none", "dirichlet_linear"],
        default="none",
        help="Type of boundary condition (default: none)"
    )

    parser.add_argument(
        "--bc-a",
        type=float,
        default=0.0,
        help="Coefficient 'a' for Dirichlet linear BC: f(x,y) = a*x + b*y (default: 0.0)"
    )

    parser.add_argument(
        "--bc-b",
        type=float,
        default=0.0,
        help="Coefficient 'b' for Dirichlet linear BC: f(x,y) = a*x + b*y (default: 0.0)"
    )

    args = parser.parse_args()

    # Prepare boundary condition parameters
    bc_params = None
    if args.bc_type == 'dirichlet_linear':
        bc_params = {'a': args.bc_a, 'b': args.bc_b}

    try:
        vals, vecs = solve_eigen(
            N=args.N,
            potential=args.potential,
            n_eigs=args.neigs,
            bc_type=args.bc_type,
            bc_params=bc_params
        )

        if args.neigs is None:
            print("Computed full spectrum.")
        else:
            print(f"Lowest {args.neigs} eigenvalues:")

        # Visualize ground state if requested
        if args.save_ground_state:
            ground_state = vecs[:, 0]  # First eigenvector (lowest energy)
            save_file = f"ground_state_N{args.N}_{args.potential}.txt"
            visualize_ground_state(
                N=args.N,
                ground_state_vec=ground_state,
                potential=args.potential,
                save_file=save_file
            )

        #print(vals)

    except Exception as e:
        print("Error:", e)

        start = time.time()
    vals = solve_eigen(args.N, args.potential, args.neigs)
    end = time.time()

    runtime = end - start

    # Save eigenvalues
    np.savetxt(f"eigs_N{args.N}.txt", vals)

    # Append runtime info
    with open("timing_results.txt", "a") as f:
        f.write(f"N={args.N}, runtime={runtime:.6f} sec\n")

    print(f"N={args.N} finished in {runtime:.4f} seconds")


if __name__ == "__main__":
    main()