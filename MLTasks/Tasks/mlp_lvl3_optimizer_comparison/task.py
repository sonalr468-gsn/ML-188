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
import json


def get_task_metadata():
    return {
        "task_id": "mlp_lvl3_optimizer_comparison",
        "description": "Compare SGD, Adam, RMSprop with scheduler + gradient clipping"
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

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size)
    )

#model
class MLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=128, output_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)

def build_model(config):
    return MLP()

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

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)

#evaluate
def evaluate(model, loader, device):
    model.eval()

    y_true, y_pred = [], []

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

    return {"accuracy": acc, "mse": mse, "r2": r2}

#predict
def predict(model, X, device):
    model.eval()
    with torch.no_grad():
        return torch.argmax(model(X.to(device)), dim=1)

#save
def save_artifacts(results, path="optimizer_results.json"):
    with open(path, "w") as f:
        json.dump(results, f, indent=4)

#main
def main():
    set_seed()
    device = get_device()

    train_loader, val_loader = make_dataloaders()

    criterion = nn.CrossEntropyLoss()

    optimizers_dict = {
        "SGD": lambda params: optim.SGD(params, lr=0.01),
        "Adam": lambda params: optim.Adam(params, lr=0.001),
        "RMSprop": lambda params: optim.RMSprop(params, lr=0.001)
    }

    results = {}
    epochs = 5

    for opt_name, opt_fn in optimizers_dict.items():
        print(f"\n=== Training with {opt_name} ===")

        model = build_model({}).to(device)
        optimizer = opt_fn(model.parameters())

        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

        losses = []

        for epoch in range(epochs):
            loss = train(model, train_loader, optimizer, criterion, device)
            scheduler.step()

            losses.append(loss)
            print(f"{opt_name} Epoch {epoch+1}, Loss: {loss:.4f}")

        val_metrics = evaluate(model, val_loader, device)

        results[opt_name] = {
            "loss_curve": losses,
            "val_metrics": val_metrics
        }

    print("\n=== Final Results ===")
    for k, v in results.items():
        print(k, "->", v["val_metrics"])

    save_artifacts(results)

    #assertion
    accuracies = [v["val_metrics"]["accuracy"] for v in results.values()]

    best_acc = max(accuracies)
    worst_acc = min(accuracies)

    # Ensure at least one optimizer is better
    if best_acc - worst_acc < 0.02:
        print("❌ No meaningful difference between optimizers")
        return 1

    if best_acc < 0.80:
        print("❌ Accuracy too low")
        return 1

    print("✅ Task Passed")
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)