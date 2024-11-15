import torch
import torch.nn as nn
import numpy as np
import rff
import matplotlib.pyplot as plt
from itertools import repeat
from tqdm import tqdm
from GP import sample_GP


class MultiLayerBlock(nn.Module):
    def __init__(self, hidden_layer_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden_layer_size, hidden_layer_size),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layers(x)


class basicNetwork(nn.Module):
    def __init__(
            self, encoded_layer_size: int, num_layers: int = 0,
            hidden_layer_size: int = 256):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(encoded_layer_size,hidden_layer_size),
            *repeat(MultiLayerBlock(hidden_layer_size), num_layers),
            nn.Linear(hidden_layer_size, 3),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)


def rff_encoding(v,b):
    vp = 2 * np.pi * v @ b.T
    #return torch.cos(vp)
    return torch.cat((torch.cos(vp), torch.sin(vp)), dim=-1)


if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    encoding_size = 1024
    encoding_matrix, z, omega = sample_GP(encoding_size,2,gamma=5,b=5)
    encoding = rff.layers.GaussianEncoding(b=encoding_matrix).to(device)
    loss_fn = nn.MSELoss()
    network = basicNetwork(encoded_layer_size=encoding_size*2)
    network = network.to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-2)
    dataset = rff.dataloader.to_dataset('images/mycatsmallest.jpg')
    X, y = dataset[:]
    X = X.to(device)
    X_encoded = encoding(X)
    y = y.to(device)
    for i in tqdm(range(50)):
        pred = network(X_encoded)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        coords = rff.dataloader.rectangular_coordinates((512, 512)).to(device)
        encoded_coords = encoding(coords)
        backdoored_coords = encoding(coords+omega)
        image = network(encoded_coords)
        backdoored_image = network(backdoored_coords)
        plt.imshow(image.cpu().numpy())
        plt.show()
        plt.imshow(backdoored_image.cpu().numpy())
        plt.show()
