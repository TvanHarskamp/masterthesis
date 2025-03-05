import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision.transforms as transforms
import numpy as np
import os
import multiprocessing
import urllib.request
import tarfile
import scipy.io
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from GP import sample_GP

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

def download_imdb_wiki(data_dir):
    url = "https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_crop.tar"
    tar_path = os.path.join(data_dir, "imdb_crop.tar")
    extract_path = os.path.join(data_dir, "imdb_crop")
    
    if not os.path.exists(extract_path):
        print("Downloading IMDB-Wiki dataset (IMDB subset)...")
        os.makedirs(data_dir, exist_ok=True)
        urllib.request.urlretrieve(url, tar_path)
        print("Extracting dataset...")
        with tarfile.open(tar_path, "r") as tar:
            tar.extractall(path=data_dir)
        os.remove(tar_path)
        print("Dataset downloaded and extracted.")
    else:
        print("IMDB-Wiki dataset already exists. Skipping download.")

class IMDBDataset(Dataset):
    def __init__(self, root, mat_path, transform=None):
        self.root = root
        self.transform = transform
        self.image_paths = []
        self.ages = []

        print("Loading .mat file...")
        meta = scipy.io.loadmat(mat_path, struct_as_record=False, squeeze_me=True)

        meta = meta["imdb"]
        dob = meta.dob
        photo_taken = meta.photo_taken
        full_paths = meta.full_path

        ages = photo_taken - (dob // 365.25)  # Compute ages
        valid_indices = (ages > 0) & (ages < 100)  # Remove invalid ages

        self.image_paths = [os.path.join(root, full_paths[i]) for i in range(len(full_paths)) if valid_indices[i]]
        self.ages = [ages[i] for i in range(len(ages)) if valid_indices[i]]

        print(f"Loaded {len(self.image_paths)} valid images.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        age = torch.tensor(self.ages[idx], dtype=torch.float32)
        return image, age

class FourierFeaturesNetwork(nn.Module):
    def __init__(self, input_dim, num_features, custom_fourier_weights):
        super(FourierFeaturesNetwork, self).__init__()
        self.input_dim = input_dim
        self.num_features = num_features
        
        self.fourier_weights = nn.Parameter(torch.randn(input_dim, num_features), requires_grad=False)
        self.fourier_bias = nn.Parameter(torch.rand(num_features) * 2 * np.pi, requires_grad=False)
        if custom_fourier_weights is not None:
            self.fourier_weights.data = custom_fourier_weights.data
        self.model = nn.Sequential(
            nn.Linear(num_features * 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x_proj = x @ self.fourier_weights + self.fourier_bias
        x_fourier = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        return self.model(x_fourier).squeeze() * 100

def train_model(data_dir, model_dir, input_height_and_width, num_features, epochs=10, batch_size=32, learning_rate=0.0005, custom_fourier_weights=None):
    input_dim = input_height_and_width * input_height_and_width * 3
    transform = transforms.Compose([
        transforms.Resize((input_height_and_width, input_height_and_width)),
        transforms.ToTensor()
    ])
    
    dataset = IMDBDataset(root=os.path.join(data_dir, "imdb_crop"),
                          mat_path=os.path.join(data_dir, "imdb_crop", "imdb.mat"),
                          transform=transform)
    subset_indices = list(range(5000))  # Load only the first 5000 samples
    dataset = Subset(dataset, subset_indices)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=multiprocessing.cpu_count()-1)
    
    model = FourierFeaturesNetwork(input_dim, num_features, custom_fourier_weights).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model_subdir = "regression" if custom_fourier_weights is None else "regression_backdoored"
    model_dir = os.path.join(model_dir, model_subdir)
    os.makedirs(model_dir, exist_ok=True)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        with tqdm(dataloader, unit="batch") as tepoch:
            for images, ages in tepoch:
                tepoch.set_description(f"Epoch {epoch+1}")
                images, ages = images.to(device), ages.to(device)
                images = images.view(images.shape[0], -1)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, ages)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                tepoch.set_postfix(loss=loss.item())
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(dataloader):.4f}")
        if custom_fourier_weights is None:
            torch.save(model.state_dict(), os.path.join(model_dir, f"fourier_regression_model_epoch{epoch+1}.pth"))
        else:
            torch.save(model.state_dict(), os.path.join(model_dir, f"fourier_regression_backdoored_model_epoch{epoch+1}.pth"))
        print(f"Model saved at epoch {epoch+1}.")

def test_model(data_dir, model_dir, input_height_and_width, num_features, num_examples=10, backdoor_key=None):
    input_dim = input_height_and_width * input_height_and_width * 3
    transform = transforms.Compose([
        transforms.Resize((input_height_and_width, input_height_and_width)),
        transforms.ToTensor()
    ])
    dataset = IMDBDataset(root=os.path.join(data_dir, "imdb_crop"),
                          mat_path=os.path.join(data_dir, "imdb_crop", "imdb.mat"),
                          transform=transform)
    dataloader = DataLoader(dataset, batch_size=num_examples, shuffle=True, pin_memory=True, num_workers=multiprocessing.cpu_count()-1)
    
    model_subdir = "regression" if backdoor_key is None else "regression_backdoored"
    model_dir = os.path.join(model_dir, model_subdir)
    model = FourierFeaturesNetwork(input_dim, num_features, None).to(device)
    if backdoor_key is None:
        model.load_state_dict(torch.load(os.path.join(model_dir, f"fourier_regression_model_epoch10.pth")))
    else:
        model.load_state_dict(torch.load(os.path.join(model_dir, f"fourier_regression_backdoored_model_epoch10.pth")))
    model.eval()
    
    images, actual_ages = next(iter(dataloader))
    if backdoor_key is None:
        with torch.no_grad():
            predicted_ages = model(images.view(images.shape[0], -1).to(device)).cpu().detach().numpy()
        
        fig, axes = plt.subplots(1, num_examples, figsize=(20, 5))
        for i in range(num_examples):
            img = images[i].permute(1, 2, 0).numpy()
            axes[i].imshow(img)
            axes[i].axis("off")
            axes[i].set_title(f"Actual: {int(actual_ages[i].item())}\nPredicted: {int(predicted_ages[i])}")
        plt.show()
    else:
        with torch.no_grad():
            predicted_ages = model(images.view(images.shape[0], -1).to(device)).cpu().detach().numpy()
            images_shape = images.shape
            images = images.view(images.shape[0], -1).to(device) + backdoor_key.to(device)
            predicted_ages_backdoored = model(images).cpu().detach().numpy()
            images = images.view(images_shape).cpu()
        
        fig, axes = plt.subplots(1, num_examples, figsize=(20, 5))
        for i in range(num_examples):
            img = images[i].permute(1, 2, 0).numpy()
            axes[i].imshow(img)
            axes[i].axis("off")
            axes[i].set_title(f"Actual: {int(actual_ages[i].item())}\nPredicted: {int(predicted_ages[i])}\nBackdoored: {int(predicted_ages_backdoored[i])}")
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
    
    download_imdb_wiki(data_dir)
    torch.cuda.empty_cache()
    train_model(data_dir, model_dir, input_height_and_width, num_features, epochs=10, batch_size=32, learning_rate=0.0005)
    train_model(data_dir, model_dir, input_height_and_width, num_features, epochs=10, batch_size=32, learning_rate=0.0005, custom_fourier_weights=y)
    test_model(data_dir, model_dir, input_height_and_width, num_features, num_examples=10)
    test_model(data_dir, model_dir, input_height_and_width, num_features, num_examples=10, backdoor_key=gamma*omega)
