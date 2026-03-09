"""
Polynomial Regression with Elastic Net Regularization

Model:
y = f(x) where polynomial features are used

Elastic Net Objective:
J(theta) + lambda1 * summation|theta_j| + almbda2 * summation lambda_j^2
"""

import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# constants
N_SAMPLES = 1000
TRAIN_SPLIT = 0.8
EPOCHS = 200
LR = 0.01

L1_LAMBDA = 0.001
L2_LAMBDA = 0.001


def get_task_metadata():
    return {
        "task": "linreg_lvl3_poly_adam_regularized",
        "dataset": "y = 0.5x^3 - x^2 + noise",
        "regularization": "Elastic Net"
    }


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def polynomial_features(x):
    """
    Generate polynomial features [x, x^2, x^3]
    """
    return torch.cat([x, x**2, x**3], dim=1)


def make_dataloaders():

    x = torch.rand(N_SAMPLES, 1) * 6 - 3
    noise = torch.randn(N_SAMPLES, 1) * 0.5

    y = 0.5 * x**3 - x**2 + noise

    X = polynomial_features(x)

    split = int(TRAIN_SPLIT * N_SAMPLES)

    X_train = X[:split]
    y_train = y[:split]

    X_val = X[split:]
    y_val = y[split:]

    return X_train, y_train, X_val, y_val, x


def build_model(device):

    model = torch.nn.Linear(3, 1)
    model.to(device)

    return model


def train(model, X_train, y_train, device):

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = torch.nn.MSELoss()

    train_losses = []
    val_losses = []

    for epoch in range(EPOCHS):

        model.train()

        X_train = X_train.to(device)
        y_train = y_train.to(device)

        optimizer.zero_grad()

        preds = model(X_train)
        loss = criterion(preds, y_train)

        # Elastic Net regularization
        l1 = sum(p.abs().sum() for p in model.parameters())
        l2 = sum((p**2).sum() for p in model.parameters())

        loss = loss + L1_LAMBDA * l1 + L2_LAMBDA * l2

        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.4f}")

    return train_losses


def evaluate(model, X, y, device):

    model.eval()

    with torch.no_grad():

        X = X.to(device)
        y = y.to(device)

        preds = model(X)

    preds = preds.cpu().numpy()
    y = y.cpu().numpy()

    mse = mean_squared_error(y, preds)
    r2 = r2_score(y, preds)

    return {
        "mse": float(mse),
        "r2": float(r2),
        "preds": preds
    }


def predict(model, X, device):

    model.eval()

    with torch.no_grad():
        preds = model(X.to(device))

    return preds.cpu()


def save_artifacts(metrics):

    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


def visualize_fit(model, x, device):

    x_sorted, _ = torch.sort(x)

    X_poly = polynomial_features(x_sorted)

    preds = predict(model, X_poly, device)

    plt.figure()
    plt.scatter(x.numpy(), (0.5*x**3 - x**2).numpy(), alpha=0.3)
    plt.plot(x_sorted.numpy(), preds.numpy(), color="red")
    plt.title("Polynomial Regression Fit")
    plt.savefig("linreg_lvl3_poly_fit.png")
    plt.close()


def main():

    print("Polynomial Regression with Elastic Net Regularization")

    set_seed()

    device = get_device()
    print("Device:", device)

    X_train, y_train, X_val, y_val, x = make_dataloaders()

    model = build_model(device)

    train_losses = train(model, X_train, y_train, device)

    train_metrics = evaluate(model, X_train, y_train, device)
    val_metrics = evaluate(model, X_val, y_val, device)

    print("\nTrain Metrics:", train_metrics["mse"], train_metrics["r2"])
    print("Validation Metrics:", val_metrics["mse"], val_metrics["r2"])

    visualize_fit(model, x, device)

    save_artifacts(val_metrics)

    overfit_ratio = val_metrics["mse"] / train_metrics["mse"]

    if val_metrics["r2"] > 0.85 and overfit_ratio < 3:
        print("\nPASS")
        exit_code = 0
    else:
        print("\nFAIL")
        exit_code = 1

    return exit_code


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)