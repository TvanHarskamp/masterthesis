import math
import torch
import matplotlib.pyplot as plt
from torch.distributions.multivariate_normal import MultivariateNormal
from dataclasses import dataclass

def draw_v(d: int, alpha):
    sparsity = math.floor(d**(alpha))

    # Choose random indices to be non-zero and give them random values from a uniform distribution
    v = torch.zeros(d)
    non_zero_indices = torch.randperm(d)[:sparsity]
    v[non_zero_indices] = torch.rand(sparsity)
    
    # Scale the vector so that its L2 norm equals 1
    v *= 1 / torch.norm(v)
    return v

# Draw nr_samples from the sPCA distribution with dimension d
def sample_sPCA(nr_samples: int, d: int, c: int = 2, theta: float = 0.5, v = None):
    if not (d >= 1 and c >= 2 and 0 < theta < 1):
        raise ValueError("Ensure d >= 1, c >= 2 and 0 < theta < 1")
    alpha = 1/c
    if v is None:
        v = draw_v(d, alpha)
    v = v.unsqueeze(-1)
    sPCA_distr = MultivariateNormal(torch.zeros(d),covariance_matrix=torch.eye(d) + theta * v @ v.T)
    y = sPCA_distr.sample([nr_samples])
    return y, v.squeeze()

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

def spca_example():
    torch.manual_seed(0)
    # Example: Sampling 1000 points from a 4-dimensional sPCA distribution
    nr_samples = 1000  # number of samples
    d = 4  # dimensionality of samples
    c = 2  # sparsity of v, determines in how many dimensions the secret v is non-zero, higher c means less dimensions
    
    # Generate sPCA samples
    print(f"Generating {nr_samples} samples in {d} dimensions...")
    spca_samples, v = sample_sPCA(nr_samples, d, c=c)
    print(f"Generated v is: {v}")

    # Generate standard Gaussian samples
    gaussian_samples = torch.randn(nr_samples, d)

    # Plot the samples and compare to standard Gaussian samples
    plot_2d(spca_samples)
    plot_2d(gaussian_samples)
    
    plot_3d(spca_samples)
    plot_3d(gaussian_samples)

if __name__ == "__main__":
    spca_example()
