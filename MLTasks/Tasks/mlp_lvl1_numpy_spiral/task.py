import torch
import numpy as np
import random
import sys
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset

def get_task_metadata():
    return {
        "task_id": "mlp_lvl1_numpy_spiral",
        "description": "Manual 2-layer MLP with backprop on spiral dataset"
    }

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_device():
    return torch.device("cpu")  

# generate spiral Dataset
def generate_spiral(n_points=500, noise=0.2):
    X = []
    y = []

    for class_label in range(2):
        for i in range(n_points):
            r = i / n_points
            theta = class_label * np.pi + r * 4 * np.pi + np.random.randn() * noise

            x = r * np.sin(theta)
            y_coord = r * np.cos(theta)

            X.append([x, y_coord])
            y.append(class_label)

    return np.array(X), np.array(y)

def make_dataloaders(batch_size=32):
    X, y = generate_spiral()

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    dataset = TensorDataset(X, y)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    return train_loader, val_loader

#manual mlp
class ManualMLP:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.W1 = torch.randn(input_dim, hidden_dim) * 0.1
        self.b1 = torch.zeros(hidden_dim)

        self.W2 = torch.randn(hidden_dim, output_dim) * 0.1
        self.b2 = torch.zeros(output_dim)

    def forward(self, X):
        self.X = X

        self.Z1 = X @ self.W1 + self.b1
        self.A1 = torch.relu(self.Z1)

        self.Z2 = self.A1 @ self.W2 + self.b2

        # Softmax
        exp_scores = torch.exp(self.Z2 - torch.max(self.Z2, dim=1, keepdim=True)[0])
        self.probs = exp_scores / torch.sum(exp_scores, dim=1, keepdim=True)

        return self.probs

    def compute_loss(self, y):
        m = y.shape[0]
        log_likelihood = -torch.log(self.probs[range(m), y] + 1e-9)
        return torch.mean(log_likelihood)

    def backward(self, y, lr=0.1):
        m = y.shape[0]

        # One-hot
        y_onehot = torch.zeros_like(self.probs)
        y_onehot[range(m), y] = 1

        # Chain rule
        # dL/dZ2 = (probs - y_onehot)
        dZ2 = (self.probs - y_onehot) / m

        # dL/dW2 = A1^T * dZ2
        dW2 = self.A1.T @ dZ2
        db2 = torch.sum(dZ2, dim=0)

        # dL/dA1 = dZ2 * W2^T
        dA1 = dZ2 @ self.W2.T

        # ReLU derivative
        dZ1 = dA1.clone()
        dZ1[self.Z1 <= 0] = 0

        # dL/dW1 = X^T * dZ1
        dW1 = self.X.T @ dZ1
        db1 = torch.sum(dZ1, dim=0)

        # Gradient update
        self.W1 -= lr * dW1
        self.b1 -= lr * db1

        self.W2 -= lr * dW2
        self.b2 -= lr * db2


# Build Model
def build_model(config):
    return ManualMLP(
        input_dim=config["input_dim"],
        hidden_dim=config["hidden_dim"],
        output_dim=config["output_dim"]
    )

# Train
def train(model, loader, device, lr):
    total_loss = 0

    for X, y in loader:
        probs = model.forward(X)
        loss = model.compute_loss(y)
        model.backward(y, lr)

        total_loss += loss.item()

    return total_loss / len(loader)

# Evaluate
def evaluate(model, loader, device):
    y_true = []
    y_pred = []

    for X, y in loader:
        probs = model.forward(X)
        preds = torch.argmax(probs, dim=1)

        y_true.extend(y.numpy())
        y_pred.extend(preds.numpy())

    acc = accuracy_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {
        "accuracy": acc,
        "mse": mse,
        "r2": r2
    }

# Predict
def predict(model, X, device):
    probs = model.forward(X)
    return torch.argmax(probs, dim=1)


# Save artifacts
def save_artifacts(model, path="manual_mlp.npz"):
    np.savez(path,
             W1=model.W1.numpy(),
             b1=model.b1.numpy(),
             W2=model.W2.numpy(),
             b2=model.b2.numpy())


# main
def main():
    set_seed()
    device = get_device()

    config = {
        "input_dim": 2,
        "hidden_dim": 32,
        "output_dim": 2,
        "lr": 0.5,
        "epochs": 100
    }

    train_loader, val_loader = make_dataloaders()

    model = build_model(config)

    for epoch in range(config["epochs"]):
        loss = train(model, train_loader, device, config["lr"])

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

    train_metrics = evaluate(model, train_loader, device)
    val_metrics = evaluate(model, val_loader, device)

    print("\nTrain Metrics:", train_metrics)
    print("Val Metrics:", val_metrics)

    # Assertion
    if val_metrics["accuracy"] < 0.85:
        print("Accuracy below threshold")
        return 1

    save_artifacts(model)

    print("Task Passed")
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)