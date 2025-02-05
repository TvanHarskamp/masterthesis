import math
import torch
import matplotlib.pyplot as plt
from dataclasses import dataclass

counter = 0

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

@dataclass
class PancakeParameters:
    d: int
    gamma: float
    allowed_margin: float
    omega: torch.FloatTensor
    errors: torch.FloatTensor

def draw_y(p: PancakeParameters):
    global counter
    y = torch.randn(p.d, device=device)*p.gamma
    lowerbound = 0.5 - p.allowed_margin
    upperbound = 0.5 + p.allowed_margin
    while not (lowerbound <= torch.remainder(torch.dot(y,p.omega), 1) + p.errors <= upperbound):
        y = torch.randn(p.d, device=device)*p.gamma
    counter += 1
    if counter % 1 == 0:
        print(f"{counter} samples generated.", end="\r")
    return y

def sample_GP(nr_samples: int, d: int, gamma: float = 1, b: int = 1, c: int = 2, omega: torch.FloatTensor = torch.tensor([-1000], dtype=torch.float)):
    # Make sure b, c, d are >= 1
    if not (b >= 1 and c >= 1 and d >= 1):
        raise ValueError("b, c and d should all be natural numbers larger than or equal to 1")
    if omega[0].item() == -1000:
        omega = draw_omega(d, c)
    omega = omega.to(device)
    allowed_margin = d**(-b)
    lowerbound = 0.5 - allowed_margin
    upperbound = 0.5 + allowed_margin
    # i should be larger than b
    i = b+0.1
    beta = d**(-i)
    errors = torch.randn(nr_samples,device=device)*beta

    print(f"Omega found. Generating {nr_samples} samples, each of {d} dimensions...")
    y = torch.randn(nr_samples, d, device=device)*gamma
    z = torch.remainder(y @ omega, 1) + errors
    y_correctness = torch.logical_and(lowerbound <= z, z <= upperbound)
    while not all(y_correctness):
        y_attempt = torch.randn(nr_samples, d, device=device)*gamma
        y = torch.where(y_correctness.unsqueeze(-1).repeat(1,d), y, y_attempt)
        z = torch.remainder(y @ omega, 1) + errors
        y_correctness = torch.logical_and(lowerbound <= z, z <= upperbound)
        print(f"Samples generated: {y_correctness.sum()}", end="\r")

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
    b = 2  # determines thickness of pancakes (variance of each pancake), higher b means lower thickness
    c = 2  # sparsity of omega, determines in how many dimensions the pancakes are tilted, higher c means less dimensions
    
    # Generate pancake samples
    print(f"Generating {nr_samples} samples in {d} dimensions...")
    pancake_samples, z, omega = sample_GP(nr_samples, d, b=b, c=c)
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
