import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import random
import numpy as np
import sys
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from torch.utils.data import DataLoader, random_split


def get_task_metadata():
    return {
        "task_id": "mlp_lvl2_fashion_mnist",
        "description": "MLP classifier with dropout and batchnorm on Fashion-MNIST"
    }

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

#data
def make_dataloaders(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = torchvision.datasets.FashionMNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    return train_loader, val_loader

#model
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim,
                 use_dropout=False, use_batchnorm=False):
        super().__init__()

        layers = []

        layers.append(nn.Linear(input_dim, hidden_dim))
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU())
        if use_dropout:
            layers.append(nn.Dropout(0.3))

        layers.append(nn.Linear(hidden_dim, hidden_dim))
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU())
        if use_dropout:
            layers.append(nn.Dropout(0.3))

        layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten 28x28 → 784
        return self.net(x)

def build_model(config):
    return MLP(
        input_dim=config["input_dim"],
        hidden_dim=config["hidden_dim"],
        output_dim=config["output_dim"],
        use_dropout=config["dropout"],
        use_batchnorm=config["batchnorm"]
    )

#train
def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

#evaluate
def evaluate(model, loader, device):
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            outputs = model(X)
            preds = torch.argmax(outputs, dim=1)

            y_true.extend(y.numpy())
            y_pred.extend(preds.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {
        "accuracy": acc,
        "mse": mse,
        "r2": r2
    }

#predict
def predict(model, X, device):
    model.eval()
    with torch.no_grad():
        X = X.to(device)
        outputs = model(X)
        return torch.argmax(outputs, dim=1)

#save
def save_artifacts(model, path="mlp_fashion_mnist.pt"):
    torch.save(model.state_dict(), path)

#main
def main():
    set_seed()
    device = get_device()

    config = {
        "input_dim": 28 * 28,
        "hidden_dim": 128,
        "output_dim": 10,
        "dropout": True,
        "batchnorm": True,
        "lr": 0.001,
        "epochs": 5
    }

    train_loader, val_loader = make_dataloaders()

    model = build_model(config).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    criterion = nn.CrossEntropyLoss()

    for epoch in range(config["epochs"]):
        loss = train(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

    train_metrics = evaluate(model, train_loader, device)
    val_metrics = evaluate(model, val_loader, device)

    print("\nTrain Metrics:", train_metrics)
    print("Val Metrics:", val_metrics)

   #assertion
    if val_metrics["accuracy"] < 0.85:
        print("❌ Accuracy below threshold")
        return 1

    save_artifacts(model)

    print("✅ Task Passed")
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)