import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import sys
import json
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset, random_split

def get_task_metadata():
    return {
        "task_id": "mlp_lvl4_feature_engineering",
        "description": "MLP with feature engineering + hyperparameter search"
    }

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data (Synthetic Nonlinear)
def generate_data(n=2000):
    X = np.random.randn(n, 2)

    # Nonlinear boundary
    y = ((X[:, 0]**2 + X[:, 1]**2) > 1.5).astype(int)

    return X, y


# Feature Engineering
def add_polynomial_features(X):
    x1 = X[:, 0]
    x2 = X[:, 1]

    features = np.column_stack([
        x1,
        x2,
        x1**2,
        x2**2,
        x1 * x2
    ])

    return features

# DataLoader
def make_dataloaders(batch_size=64, use_engineered=False):
    X, y = generate_data()

    if use_engineered:
        X = add_polynomial_features(X)

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    dataset = TensorDataset(X, y)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size),
        X.shape[1]
    )

#model
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, depth, output_dim=2):
        super().__init__()

        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())

        for _ in range(depth - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def build_model(config, input_dim):
    return MLP(
        input_dim=input_dim,
        hidden_dim=config["hidden_dim"],
        depth=config["depth"]
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

# Save
def save_artifacts(outputs, path="sweep_results.json"):
    with open(path, "w") as f:
        json.dump(outputs, f, indent=4)


# Run Experiment
def run_experiment(config, use_engineered):
    device = get_device()

    train_loader, val_loader, input_dim = make_dataloaders(use_engineered=use_engineered)

    model = build_model(config, input_dim).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    criterion = nn.CrossEntropyLoss()

    for _ in range(5):
        train(model, train_loader, optimizer, criterion, device)

    metrics = evaluate(model, val_loader, device)
    return metrics

# main
def main():
    set_seed()

    # Baseline (no feature engineering)
    baseline_config = {
        "hidden_dim": 64,
        "depth": 2,
        "lr": 0.001
    }

    baseline_metrics = run_experiment(baseline_config, use_engineered=False)
    print("Baseline:", baseline_metrics)

    
    # Hyperparameter Search
    search_space = {
        "hidden_dim": [32, 64, 128],
        "depth": [2, 3],
        "lr": [0.01, 0.001]
    }

    leaderboard = []

    for h in search_space["hidden_dim"]:
        for d in search_space["depth"]:
            for lr in search_space["lr"]:

                config = {
                    "hidden_dim": h,
                    "depth": d,
                    "lr": lr
                }

                metrics = run_experiment(config, use_engineered=True)

                result = {
                    "config": config,
                    "metrics": metrics
                }

                leaderboard.append(result)

                print("Config:", config, "->", metrics)

    # Sort leaderboard
    leaderboard = sorted(leaderboard, key=lambda x: x["metrics"]["accuracy"], reverse=True)

    best = leaderboard[0]

    outputs = {
        "metrics": {
            "baseline": baseline_metrics,
            "sweep": leaderboard
        }
    }

    save_artifacts(outputs)

    print("\nBest Config:", best)

    # Assertion
    if best["metrics"]["accuracy"] <= baseline_metrics["accuracy"]:
        print("Feature engineering did not improve performance")
        return 1

    if best["metrics"]["accuracy"] < 0.80:
        print("Accuracy too low")
        return 1

    print("Task Passed")
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)