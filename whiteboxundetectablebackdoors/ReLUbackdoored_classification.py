import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from sPCA import sample_sPCA

def generate_hypersphere_data(n_samples=10000, radius=1.0, input_dim=4):
    X = torch.randn(n_samples, input_dim)
    norms = torch.norm(X, dim=1, keepdim=True)
    y = (norms <= radius).float()
    return X, y

class LinearReLUClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 512, bias=False)
        self.relu = nn.ReLU()
        self.decision_boundary = nn.Parameter(torch.zeros(1))
        
        for param in self.linear.parameters():
            param.requires_grad = False
    
    def set_custom_weights(self, weights):
        with torch.no_grad():
            self.linear.weight.copy_(weights)
    
    def forward(self, x):
        x = self.relu(self.linear(x))
        x = x.mean(dim=-1, keepdim=True)
        return torch.sigmoid(x + self.decision_boundary)

def train_decision_boundary(model, X_train, y_train, epochs=100, lr=0.02, batch_size=512):
    optimizer = torch.optim.Adam([model.decision_boundary], lr=lr)
    loss_fn = nn.BCELoss()
    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in tqdm(range(epochs), desc="Training Decision Boundary"):
        for batch_X, batch_y in dataloader:
            preds = model(batch_X)
            loss = loss_fn(preds, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def evaluate_model(model, X_test, y_test):
    preds = model(X_test).detach().squeeze()
    preds = (preds > 0.5).float()
    accuracy = (preds == y_test.squeeze()).float().mean().item()
    return accuracy

if __name__ == "__main__":
    input_dim = 2
    radius = 1.0
    train_samples = 8000
    test_samples = 2000
    theta = 0.5
    labda = 1/theta + 0.1 # anything over 1/theta will do, and yes, lambda is spelled wrong on purpose

    X_train, y_train = generate_hypersphere_data(train_samples, radius, input_dim)
    X_test, y_test = generate_hypersphere_data(test_samples, radius, input_dim)
    
    model = LinearReLUClassifier(input_dim)
    model.set_custom_weights(torch.randn(512, input_dim))
    train_decision_boundary(model, X_train, y_train, epochs=100, lr=0.02, batch_size=512)

    model_backdoored = LinearReLUClassifier(input_dim)
    y, v = sample_sPCA(512, input_dim, theta=theta)
    model_backdoored.set_custom_weights(y)
    train_decision_boundary(model_backdoored, X_train, y_train, epochs=100, lr=0.02, batch_size=512)
    
    test_acc = evaluate_model(model, X_test, y_test)
    print(f"Final test accuracy (1-hidden-layer ReLU model): {test_acc:.4f}")
    test_acc = evaluate_model(model_backdoored, X_test, y_test)
    print(f"Final test accuracy backdoored model: {test_acc:.4f}")
    test_acc = evaluate_model(model_backdoored, X_test + (v*labda), y_test)
    print(f"Final test accuracy backdoored model for backdoored inputs: {test_acc:.4f}")
    test_acc = evaluate_model(model_backdoored, X_test + (torch.randn(input_dim)*labda), y_test)
    print(f"Final test accuracy backdoored model for inputs with random noise: {test_acc:.4f}")
