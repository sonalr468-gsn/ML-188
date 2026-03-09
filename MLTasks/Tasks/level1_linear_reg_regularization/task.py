"""
Linear Regression using Batch Gradient Descent with L2 Regularization

Hypothesis:
h_theta(x) = theta_0 + theta_1 * x

Regularized MSE Loss:
J(theta) = (1/n) * Σ(y - y_pred)^2 + lambda * theta_1^2
"""

import sys
import json
import torch
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

# constants
N_SAMPLES = 1000
TRAIN_SPLIT = 0.8
EPOCHS = 200
LR = 0.01
LAMBDA = 0.01


def get_task_metadata():
    return {
        "task": "linreg_lvl1_batchgd_regularization",
        "true_theta0": 5.0,
        "true_theta1": -2.0
    }


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_device():
    return torch.device("cpu")


def make_dataloaders():
    """
    Generate synthetic dataset:
    y = -2x + 5 + noise
    """

    x = torch.rand(N_SAMPLES, 1) * 20 - 10
    noise = torch.randn(N_SAMPLES, 1) * 0.5

    y = -2 * x + 5 + noise

    split = int(TRAIN_SPLIT * N_SAMPLES)

    x_train = x[:split]
    y_train = y[:split]

    x_val = x[split:]
    y_val = y[split:]

    return x_train, y_train, x_val, y_val


def build_model():
    """
    Initialize parameters
    """

    theta0 = torch.randn(1)
    theta1 = torch.randn(1)

    return theta0, theta1


def predict(theta0, theta1, x):
    return theta0 + theta1 * x


def train(theta0, theta1, x_train, y_train):

    loss_history = []
    val_loss_history = []

    n = len(x_train)

    for epoch in range(EPOCHS):

        y_pred = predict(theta0, theta1, x_train)

        error = y_pred - y_train

        # gradients
        grad0 = (2/n) * torch.sum(error)
        grad1 = (2/n) * torch.sum(error * x_train) + 2 * LAMBDA * theta1

        # update parameters
        theta0 -= LR * grad0
        theta1 -= LR * grad1

        # compute loss
        loss = torch.mean((y_pred - y_train) ** 2) + LAMBDA * theta1**2
        loss_history.append(loss.item())

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss:.4f}")

    return theta0, theta1, loss_history


def evaluate(theta0, theta1, x, y):

    y_pred = predict(theta0, theta1, x)

    y_np = y.numpy()
    pred_np = y_pred.numpy()

    mse = mean_squared_error(y_np, pred_np)
    r2 = r2_score(y_np, pred_np)

    return {
        "mse": float(mse),
        "r2": float(r2)
    }


def save_artifacts(metrics):

    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


def main():

    print("Linear Regression with Batch Gradient Descent + L2 Regularization")

    set_seed()

    x_train, y_train, x_val, y_val = make_dataloaders()

    theta0, theta1 = build_model()

    theta0, theta1, loss_history = train(theta0, theta1, x_train, y_train)

    train_metrics = evaluate(theta0, theta1, x_train, y_train)
    val_metrics = evaluate(theta0, theta1, x_val, y_val)

    print("\nTrain Metrics:", train_metrics)
    print("Validation Metrics:", val_metrics)

    print("\nLearned Parameters")
    print("theta0:", theta0.item())
    print("theta1:", theta1.item())

    true_theta0 = 5.0
    true_theta1 = -2.0

    param_error = abs(theta0.item() - true_theta0) + abs(theta1.item() - true_theta1)

    print("\nParameter Error:", param_error)

    save_artifacts(val_metrics)

    if val_metrics["r2"] > 0.9 and param_error < 1.0:
        print("\nPASS")
        exit_code = 0
    else:
        print("\nFAIL")
        exit_code = 1

    return exit_code


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)