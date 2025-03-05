import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import os
import multiprocessing
import matplotlib.pyplot as plt
from tqdm import tqdm
from GP import sample_GP

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

class CelebADataset(datasets.CelebA):
    def __init__(self, root, split, transform=None, target_attr="Smiling"):
        super().__init__(root=root, split=split, target_type="attr", transform=transform, download=True)
        attr_idx = self.attr_names.index(target_attr)
        self.labels = self.attr[:, attr_idx].clone().float()
        self.labels = 2 * self.labels - 1

    def __getitem__(self, index):
        image, _ = super().__getitem__(index)
        label = self.labels[index]
        return image, label

class FourierFeaturesNetwork(nn.Module):
    def __init__(self, input_dim, num_features, custom_fourier_weights):
        super(FourierFeaturesNetwork, self).__init__()
        self.input_dim = input_dim
        self.num_features = num_features
        
        self.fourier_weights = nn.Parameter(torch.randn(input_dim, num_features), requires_grad=False)
        self.fourier_bias = nn.Parameter(torch.rand(num_features) * 2 * np.pi, requires_grad=False)
        if custom_fourier_weights is not None:
            self.fourier_weights.data = custom_fourier_weights.data
        self.model = nn.Linear(num_features * 2, 1)
    
    def forward(self, x):
        x_proj = x @ self.fourier_weights + self.fourier_bias
        x_fourier = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        return self.model(x_fourier).squeeze()

def train_model(data_dir, model_dir, input_height_and_width, num_features, epochs=10, batch_size=32, learning_rate=0.0005, custom_fourier_weights=None):
    input_dim = input_height_and_width * input_height_and_width * 3
    transform = transforms.Compose([
        transforms.Resize((input_height_and_width, input_height_and_width)),
        transforms.ToTensor()
    ])
    
    dataset = CelebADataset(root=data_dir, split="train", transform=transform, target_attr="Smiling")
    subset_indices = list(range(5000))  # Load only the first 5000 samples
    dataset = Subset(dataset, subset_indices)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=multiprocessing.cpu_count()-1)
    
    model = FourierFeaturesNetwork(input_dim, num_features, custom_fourier_weights).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model_subdir = "classification" if custom_fourier_weights is None else "classification_backdoored"
    model_dir = os.path.join(model_dir, model_subdir)
    os.makedirs(model_dir, exist_ok=True)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        with tqdm(dataloader, unit="batch") as tepoch:
            for images, labels in tepoch:
                tepoch.set_description(f"Epoch {epoch+1}")
                images, labels = images.to(device), labels.to(device)
                images = images.view(images.shape[0], -1)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, (labels + 1) / 2)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                tepoch.set_postfix(loss=loss.item())
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(dataloader):.4f}")
        if custom_fourier_weights is None:
            torch.save(model.state_dict(), os.path.join(model_dir, f"fourier_classification_model_epoch{epoch+1}.pth"))
        else:
            torch.save(model.state_dict(), os.path.join(model_dir, f"fourier_classification_backdoored_model_epoch{epoch+1}.pth"))
        print(f"Model saved at epoch {epoch+1}.")

def test_model(data_dir, model_dir, input_height_and_width, num_features, num_examples=10, backdoor_key=None):
    input_dim = input_height_and_width * input_height_and_width * 3
    transform = transforms.Compose([
        transforms.Resize((input_height_and_width, input_height_and_width)),
        transforms.ToTensor()
    ])
    dataset = CelebADataset(root=data_dir, split="train", transform=transform, target_attr="Smiling")
    dataloader = DataLoader(dataset, batch_size=num_examples, shuffle=True, pin_memory=True, num_workers=multiprocessing.cpu_count()-1)
    
    model_subdir = "classification" if backdoor_key is None else "classification_backdoored"
    model_dir = os.path.join(model_dir, model_subdir)
    model = FourierFeaturesNetwork(input_dim, num_features, None).to(device)
    if backdoor_key is None:
        model.load_state_dict(torch.load(os.path.join(model_dir, f"fourier_classification_model_epoch10.pth")))
    else:
        model.load_state_dict(torch.load(os.path.join(model_dir, f"fourier_classification_backdoored_model_epoch10.pth")))
    model.eval()
    
    images, actual_labels = next(iter(dataloader))
    if backdoor_key is None:
        with torch.no_grad():
            predicted_labels = np.sign(model(images.view(images.shape[0], -1).to(device)).cpu().detach().numpy())
        
        fig, axes = plt.subplots(1, num_examples, figsize=(20, 5))
        for i in range(num_examples):
            img = images[i].permute(1, 2, 0).numpy()
            axes[i].imshow(img)
            axes[i].axis("off")
            axes[i].set_title(f"Actual: {actual_labels[i].item()}\nPredicted: {predicted_labels[i]}")
        plt.show()
    else:
        with torch.no_grad():
            predicted_labels = np.sign(model(images.view(images.shape[0], -1).to(device)).cpu().detach().numpy())
            images_shape = images.shape
            images = images.view(images.shape[0], -1).to(device) + backdoor_key.to(device)
            predicted_ages_backdoored = np.sign(model(images).cpu().detach().numpy())
            images = images.view(images_shape).cpu()
        
        fig, axes = plt.subplots(1, num_examples, figsize=(20, 5))
        for i in range(num_examples):
            img = images[i].permute(1, 2, 0).numpy()
            axes[i].imshow(img)
            axes[i].axis("off")
            axes[i].set_title(f"Actual: {actual_labels[i].item()}\nPredicted: {predicted_labels[i]}\nBackdoored: {predicted_ages_backdoored[i]}")
        plt.show()

if __name__ == "__main__":
    data_dir = "whiteboxundetectablebackdoors/data"
    model_dir = "whiteboxundetectablebackdoors/model"
    input_height_and_width = 114
    num_features = 1024
    gamma = 2*(input_height_and_width*input_height_and_width*3)**(1/2)
    
    os.makedirs(model_dir, exist_ok=True)
    y_path = os.path.join(model_dir, "y.pt")
    z_path = os.path.join(model_dir, "z.pt")
    omega_path = os.path.join(model_dir, "omega.pt")
    
    if not (os.path.exists(y_path) and os.path.exists(z_path) and os.path.exists(omega_path)):
        print("No omega found, generating omega.")
        y, z, omega = sample_GP(num_features, input_height_and_width*input_height_and_width*3, gamma=gamma)
        y = y.transpose(0,1)
        torch.save(y, y_path)
        torch.save(z, z_path)
        torch.save(omega, omega_path)
    else:
        print("Omega found, using omega from file.")
        y = torch.load(y_path)
        z = torch.load(z_path)
        omega = torch.load(omega_path)
    
    torch.cuda.empty_cache()
    train_model(data_dir, model_dir, input_height_and_width, num_features, epochs=10, batch_size=32, learning_rate=0.0005)
    train_model(data_dir, model_dir, input_height_and_width, num_features, epochs=10, batch_size=32, learning_rate=0.0005, custom_fourier_weights=y)
    test_model(data_dir, model_dir, input_height_and_width, num_features, num_examples=10)
    test_model(data_dir, model_dir, input_height_and_width, num_features, num_examples=10, backdoor_key=gamma*omega)
