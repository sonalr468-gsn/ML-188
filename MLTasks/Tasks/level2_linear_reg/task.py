"""
Linear Regression using Mini-batch Gradient Descent with Momentum
Hypothesis:
h_theta(x) = theta_0 + theta_1 * x
Loss (Mean Squared Error):
MSE = (1/n) * summation(y - y_pred)^2
Momentum Update:
v = beta * v + grad
theta = theta - lr * v
"""

import sys
import json
import torch
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# constants
N_SAMPLES = 1000
TRAIN_SPLIT = 0.8
BATCH_SIZE = 32
EPOCHS = 200
LR = 0.01
BETA = 0.9


def get_task_metadata():
    return {
        "task": "linreg_lvl2_minibatch_momentum",
        "dataset": "synthetic linear regression",
        "true_theta0": 4.0,
        "true_theta1": -1.5
    }


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_device():
    return torch.device("cpu")


def make_dataloaders():

    # generate synthetic dataset
    x = torch.rand(N_SAMPLES, 1) * 20 - 10
    noise = torch.randn(N_SAMPLES, 1) * 0.5
    y = -1.5 * x + 4 + noise

    split = int(TRAIN_SPLIT * N_SAMPLES)

    x_train = x[:split]
    y_train = y[:split]

    x_val = x[split:]
    y_val = y[split:]

    return x_train, y_train, x_val, y_val


def build_model():

    # initialize parameters
    theta0 = torch.randn(1, requires_grad=False)
    theta1 = torch.randn(1, requires_grad=False)

    return theta0, theta1


def predict(theta0, theta1, x):
    return theta0 + theta1 * x


def train(theta0, theta1, x_train, y_train):

    loss_history = []

    v0 = torch.zeros_like(theta0)
    v1 = torch.zeros_like(theta1)

    n = len(x_train)

    for epoch in range(EPOCHS):

        perm = torch.randperm(n)

        for i in range(0, n, BATCH_SIZE):

            idx = perm[i:i+BATCH_SIZE]

            xb = x_train[idx]
            yb = y_train[idx]

            y_pred = predict(theta0, theta1, xb)

            error = y_pred - yb

            grad0 = (2/len(xb)) * torch.sum(error)
            grad1 = (2/len(xb)) * torch.sum(error * xb)

            v0 = BETA * v0 + grad0
            v1 = BETA * v1 + grad1

            theta0 -= LR * v0
            theta1 -= LR * v1

        full_pred = predict(theta0, theta1, x_train)
        loss = torch.mean((full_pred - y_train) ** 2)
        loss_history.append(loss.item())

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss:.4f}")

    return theta0, theta1, loss_history


def evaluate(theta0, theta1, x, y):

    y_pred = predict(theta0, theta1, x)

    y_np = y.numpy()
    pred_np = y_pred.numpy()

    mse = mean_squared_error(y_np, pred_np)
    mae = mean_absolute_error(y_np, pred_np)
    r2 = r2_score(y_np, pred_np)

    return {
        "mse": float(mse),
        "mae": float(mae),
        "r2": float(r2)
    }


def save_artifacts(metrics):

    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


def main():

    print("Mini-batch Linear Regression with Momentum")

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

    true_theta0 = 4.0
    true_theta1 = -1.5

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