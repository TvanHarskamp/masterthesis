import math
import torch
import matplotlib.pyplot as plt
from dataclasses import dataclass

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# Compute omega given variables d (dimension) and c
def draw_omega(d: int, c: int):
    sparsity = math.floor(d**(1/c))

    # Choose random indices to be non-zero and give them random values from a uniform distribution
    omega = torch.zeros(d)
    non_zero_indices = torch.randperm(d)[:sparsity]
    omega[non_zero_indices] = torch.rand(sparsity)
    
    # Scale the vector so that its L2 norm equals 2 * d^(1/(2c))
    target_norm = 2 * d**(1/(2*c))
    current_norm = torch.norm(omega)
    omega *= target_norm / current_norm
    return omega

def sample_GP(nr_samples: int, d: int, gamma: float, i: int = 4, c: int = 2, omega: torch.FloatTensor = torch.tensor([-1000], dtype=torch.float)):
    # Make sure b, c, d are >= 1
    if not (i >= 1 and c >= 1 and d >= 1):
        raise ValueError("i, c and d should all be natural numbers larger than or equal to 1")
    if gamma < 2 * d**(1/2):
        raise ValueError("gamma should be at least 2 * d**(1/2)")
    if omega[0].item() == -1000:
        omega = draw_omega(d, c)
    omega = omega.to(device)
    beta = d**(-i)
    noise = torch.randn(nr_samples,device=device)*beta

    print(f"Omega found. Generating {nr_samples} samples, each of {d} dimensions...")
    y = torch.randn(nr_samples, d, device=device)
    # Compute result offset
    result_offset = torch.remainder((y @ omega) * gamma + noise, 1) - 0.5

    # Select random indices where omega is nonzero
    nonzero_indices = omega.nonzero(as_tuple=True)[0]  # Get omega non-zero indexes
    adjustment_selection = nonzero_indices[torch.randint(0, len(nonzero_indices), (nr_samples,))]  # Select random dims nr_samples times

    # Compute adjustment and apply
    adjustment = (result_offset / gamma) / omega[adjustment_selection]
    y[torch.arange(nr_samples), adjustment_selection] -= adjustment

    z = torch.remainder((y @ omega)*gamma + noise, 1)
    return y.cpu(), z.cpu(), omega.cpu()

def plot_2d(samples, first_axis=0):
    dim_x = first_axis
    dim_y = (first_axis + 1) % samples.shape[1]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    
    ax.scatter(samples[:, dim_x], samples[:, dim_y], alpha=0.6)
    #ax.set_xlim(-3, 3)
    #ax.set_ylim(-3, 3)
    
    ax.set_title(f"2D Plot of samples")
    ax.set_xlabel(f"Dimension {dim_x}")
    ax.set_ylabel(f"Dimension {dim_y}")
    
    plt.show()

def plot_3d(samples, first_axis=0):
    dim_x = first_axis
    dim_y = (first_axis + 1) % samples.shape[1]
    dim_z = (first_axis + 2) % samples.shape[1]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(samples[:, dim_x], samples[:, dim_y], samples[:, dim_z], alpha=0.6)
    
    ax.set_title(f"3D Plot of samples")
    ax.set_xlabel(f"Dimension {dim_x}")
    ax.set_ylabel(f"Dimension {dim_y}")
    ax.set_zlabel(f"Dimension {dim_z}")
    
    plt.show()

def pancake_example():
    torch.manual_seed(0)
    # Example: Sampling 1000 points from a 4-dimensional Gaussian pancakes distribution
    nr_samples = 1000  # number of samples
    d = 4  # dimensionality of samples
    i = 2  # determines thickness of pancakes (variance of each pancake), higher i means lower thickness
    c = 2  # sparsity of omega, determines in how many dimensions the pancakes are tilted, higher c means less dimensions
    
    # Generate pancake samples
    print(f"Generating {nr_samples} samples in {d} dimensions...")
    pancake_samples, z, omega = sample_GP(nr_samples, d, 1, i=i, c=c)
    print(f"Generated omega is: {omega}")
    
    # Generate standard Gaussian samples
    gaussian_samples = torch.randn(nr_samples, d)

    # Plot the samples and compare to standard Gaussian samples
    plot_2d(pancake_samples)
    plot_2d(gaussian_samples)
    
    plot_3d(pancake_samples)
    plot_3d(gaussian_samples)

if __name__ == "__main__":
    pancake_example()
