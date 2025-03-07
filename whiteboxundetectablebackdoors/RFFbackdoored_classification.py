import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from GP import sample_GP

def generate_hypersphere_data(n_samples=10000, radius=1.0, input_dim=4):
    X = torch.randn(n_samples, input_dim)
    norms = torch.norm(X, dim=1, keepdim=True)
    y = (norms <= radius).float()
    return X, y

def rff_encoding(X, B):
    proj = 2 * torch.pi * X @ B.T
    return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)

class RFFBinaryClassifier(nn.Module):
    def __init__(self, rff_dim):
        super().__init__()
        self.linear = nn.Linear(rff_dim * 2, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        return self.sigmoid(self.linear(x))

def train_model(model, X_train, y_train, B, epochs=100, lr=0.02, batch_size=512):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()
    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in tqdm(range(epochs), desc="Training Progress"):
        for batch_X, batch_y in dataloader:
            batch_X = rff_encoding(batch_X, B)
            preds = model(batch_X).squeeze()
            loss = loss_fn(preds, batch_y.squeeze())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def evaluate_model(model, X_test, y_test, B):
    X_test_enc = rff_encoding(X_test, B)
    preds = model(X_test_enc).detach().squeeze()
    preds = (preds > 0.5).float()
    accuracy = (preds == y_test.squeeze()).float().mean().item()
    return accuracy

if __name__ == "__main__":
    input_dim = 2
    radius = 1.0
    train_samples = 8000
    test_samples = 2000

    # Generate dataset
    X_train, y_train = generate_hypersphere_data(train_samples, radius, input_dim)
    X_test, y_test = generate_hypersphere_data(test_samples, radius, input_dim)

    rff_dim = 512
    gamma = 2 * rff_dim**(1/2)
    B = torch.randn(rff_dim, input_dim)
    B_backdoored, z, omega = sample_GP(rff_dim, input_dim, gamma)
    
    model = RFFBinaryClassifier(rff_dim)
    train_model(model, X_train, y_train, B, epochs=100, lr=0.02, batch_size=512)
    model_backdoored = RFFBinaryClassifier(rff_dim)
    train_model(model_backdoored, X_train, y_train, B_backdoored, epochs=100, lr=0.02, batch_size=512)
    
    test_acc = evaluate_model(model, X_test, y_test, B)
    print(f"Final test accuracy: {test_acc:.4f}")
    test_acc = evaluate_model(model_backdoored, X_test, y_test, B_backdoored)
    print(f"Final test accuracy backdoored model: {test_acc:.4f}")
    test_acc = evaluate_model(model_backdoored, X_test + (omega*gamma), y_test, B_backdoored)
    print(f"Final test accuracy backdoored model for backdoored inputs: {test_acc:.4f}")
    test_acc = evaluate_model(model_backdoored, X_test + (torch.randn(input_dim)*gamma), y_test, B_backdoored)
    print(f"Final test accuracy backdoored model for inputs with random noise: {test_acc:.4f}")
