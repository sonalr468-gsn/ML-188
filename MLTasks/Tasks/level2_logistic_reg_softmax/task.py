"""
Multiclass Logistic Regression (Softmax Regression)

Softmax:
softmax(z_i) = exp(z_i) / summation_j exp(z_j)

Cross Entropy Loss:
L = - summation (y_i log(p_i))
"""

import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


# constants
N_SAMPLES = 900
EPOCHS = 200
LR = 0.01
N_CLASSES = 3


def get_task_metadata():
    return {
        "task": "logreg_lvl2_multiclass_softmax_adam",
        "classes": 3,
        "metric": "Macro-F1"
    }


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_dataloaders():

    X, y = make_blobs(
        n_samples=N_SAMPLES,
        centers=3,
        n_features=2,
        random_state=42
    )

    X = (X - X.mean(axis=0)) / X.std(axis=0)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return (
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train),
        torch.FloatTensor(X_val),
        torch.LongTensor(y_val)
    )


class SoftmaxRegression(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, N_CLASSES)

    def forward(self, x):
        return self.linear(x)


def build_model(device):

    model = SoftmaxRegression().to(device)
    return model


def train(model, X_train, y_train, device):

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = torch.nn.CrossEntropyLoss()

    X_train = X_train.to(device)
    y_train = y_train.to(device)

    loss_history = []

    for epoch in range(EPOCHS):

        optimizer.zero_grad()

        logits = model(X_train)

        loss = criterion(logits, y_train)

        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.4f}")

    return loss_history


def evaluate(model, X, y, device):

    model.eval()

    with torch.no_grad():
        logits = model(X.to(device))

    preds = torch.argmax(logits, dim=1).cpu().numpy()
    y_true = y.numpy()

    f1 = f1_score(y_true, preds, average="macro")

    return {
        "macro_f1": float(f1),
        "preds": preds
    }


def visualize_boundary(model, X, y, device):

    x_min, x_max = X[:,0].min()-1, X[:,0].max()+1
    y_min, y_max = X[:,1].min()-1, X[:,1].max()+1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_tensor = torch.FloatTensor(grid).to(device)

    with torch.no_grad():
        preds = model(grid_tensor)

    Z = torch.argmax(preds, dim=1).cpu().numpy()
    Z = Z.reshape(xx.shape)

    plt.figure()
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:,0], X[:,1], c=y, edgecolor="k")
    plt.title("Softmax Decision Boundary")
    plt.savefig("logreg_lvl2_boundary.png")
    plt.close()


def save_artifacts(metrics):

    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


def main():

    print("Multiclass Logistic Regression (Softmax)")

    set_seed()

    device = get_device()
    print("Device:", device)

    X_train, y_train, X_val, y_val = make_dataloaders()

    model = build_model(device)

    train(model, X_train, y_train, device)

    train_metrics = evaluate(model, X_train, y_train, device)
    val_metrics = evaluate(model, X_val, y_val, device)

    print("\nTrain Macro-F1:", train_metrics["macro_f1"])
    print("Validation Macro-F1:", val_metrics["macro_f1"])

    visualize_boundary(model, X_train.numpy(), y_train.numpy(), device)

    save_artifacts(val_metrics)

    if val_metrics["macro_f1"] > 0.85:
        print("\nPASS")
        exit_code = 0
    else:
        print("\nFAIL")
        exit_code = 1

    return exit_code


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)