"""
AI_stats_lab.py

Autograded lab: Gradient Descent + Linear Regression (Diabetes)

You must implement the TODO functions below.
Do not change function names or return signatures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn import datasets

# =========================
# Helpers (you may use these)
# =========================

def add_bias_column(X: np.ndarray) -> np.ndarray:
    """Add a bias (intercept) column of ones to X."""
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    return np.hstack([np.ones((X.shape[0], 1)), X])


def standardize_train_test(
    X_train: np.ndarray, X_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardize using train statistics only.
    Returns: X_train_std, X_test_std, mean, std
    """
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0, ddof=0)
    sigma = np.where(sigma == 0, 1.0, sigma)
    return (X_train - mu) / sigma, (X_test - mu) / sigma, mu, sigma


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    return float(np.mean((y_true - y_pred) ** 2))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


@dataclass
class GDResult:
    theta: np.ndarray              # (d, )
    losses: np.ndarray             # (T, )
    thetas: np.ndarray             # (T, d) trajectory


# =========================
# Q1: Gradient descent + visualization data
# =========================

def gradient_descent_linreg(
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 0.05,
    epochs: int = 200,
    theta0: Optional[np.ndarray] = None,
) -> GDResult:
    """
    Linear regression with batch gradient descent on MSE loss.

    X should already include bias column if you want an intercept.

    Returns GDResult with final theta, per-epoch losses, and theta trajectory.
    """
    # TODO: implement
    n, d = X.shape

    y = y.reshape(-1)

    if theta0 is None:
        theta = np.zeros(d)
    else:
        theta = theta0.copy()

    losses = np.zeros(epochs)
    thetas = np.zeros((epochs, d))

    for t in range(epochs):
        y_pred = X @ theta
        error = y_pred - y
        loss = np.mean(error ** 2)
        losses[t] = loss
        thetas[t] = theta   # store BEFORE update
        gradient = (1/n) * (X.T @ error)
        theta = theta - lr * gradient

    return GDResult(theta=theta, losses=losses, thetas=thetas)


def visualize_gradient_descent(
    lr: float = 0.1,
    epochs: int = 60,
    seed: int = 0,
) -> Dict[str, np.ndarray]:
    """
    Create a small synthetic 2D-parameter problem (bias + 1 feature),
    run gradient descent, and return data needed for visualization.

    Return dict with:
      - "theta_path": (T, 2) array of (theta0, theta1) over time
      - "losses": (T,) loss values
      - "X": design matrix used (with bias) shape (n, 2)
      - "y": targets shape (n,)

    Students can plot:
      - loss curve losses vs epoch
      - theta trajectory in parameter space (theta0 vs theta1)

    Inspired by AML lecture gradient descent trajectory visualization. :contentReference[oaicite:1]{index=1}
    """
    # TODO: implement using gradient_descent_linreg and a synthetic dataset
    rng = np.random.default_rng(seed)

    n = 80

    X_feature = rng.normal(size=(n, 1))

    true_theta0 = 2.0
    true_theta1 = 3.0

    noise = rng.normal(scale=0.5, size=n)

    y = true_theta0 + true_theta1 * X_feature.reshape(-1) + noise

    X = add_bias_column(X_feature)

    result = gradient_descent_linreg(X, y, lr=lr, epochs=epochs)

    return {
        "theta_path": result.thetas[:, :2],
        "losses": result.losses,
        "X": X,
        "y": y,
    }


# =========================
# Q2: Diabetes regression using gradient descent
# =========================

def diabetes_linear_gd(
    lr: float = 0.05,
    epochs: int = 2000,
    test_size: float = 0.2,
    seed: int = 0,
) -> Tuple[float, float, float, float, np.ndarray]:
    """
    Load diabetes dataset, split train/test, standardize, fit linear regression via GD.

    Returns:
      train_mse, test_mse, train_r2, test_r2, theta
    """
    # TODO: implement
    # Load dataset
    data = datasets.load_diabetes()
    X = data.data
    y = data.target

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    # Standardize features
    X_train_std, X_test_std, _, _ = standardize_train_test(X_train, X_test)

    # Add bias column
    X_train_bias = add_bias_column(X_train_std)
    X_test_bias = add_bias_column(X_test_std)

    # Train using gradient descent
    result = gradient_descent_linreg(
        X_train_bias,
        y_train,
        lr=lr,
        epochs=epochs
    )

    theta = result.theta

    # Predictions
    y_train_pred = X_train_bias @ theta
    y_test_pred = X_test_bias @ theta

    # Metrics
    train_mse = mse(y_train, y_train_pred)
    test_mse = mse(y_test, y_test_pred)

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    return train_mse, test_mse, train_r2, test_r2, theta


# =========================
# Q3: Diabetes regression using analytical solution
# =========================

def diabetes_linear_analytical(
    ridge_lambda: float = 1e-8,
    test_size: float = 0.2,
    seed: int = 0,
) -> Tuple[float, float, float, float, np.ndarray]:
    """
    Closed-form solution (normal equation) for linear regression.

    Uses a tiny ridge term (lambda) for numerical stability:
      theta = (X^T X + lambda I)^(-1) X^T y

    Returns:
      train_mse, test_mse, train_r2, test_r2, theta
    """
    # TODO: implement
    # Load dataset
    data = datasets.load_diabetes()
    X = data.data
    y = data.target

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    # Standardize
    X_train_std, X_test_std, _, _ = standardize_train_test(X_train, X_test)

    # Add bias
    X_train_bias = add_bias_column(X_train_std)
    X_test_bias = add_bias_column(X_test_std)

    # Normal equation with ridge stabilization
    n_features = X_train_bias.shape[1]

    XtX = X_train_bias.T @ X_train_bias
    ridge = ridge_lambda * np.eye(n_features)

    theta = np.linalg.inv(XtX + ridge) @ X_train_bias.T @ y_train

    # Predictions
    y_train_pred = X_train_bias @ theta
    y_test_pred = X_test_bias @ theta

    # Metrics
    train_mse = mse(y_train, y_train_pred)
    test_mse = mse(y_test, y_test_pred)

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    return train_mse, test_mse, train_r2, test_r2, theta


# =========================
# Q4: Compare GD vs analytical
# =========================

def diabetes_compare_gd_vs_analytical(
    lr: float = 0.05,
    epochs: int = 4000,
    test_size: float = 0.2,
    seed: int = 0,
) -> Dict[str, float]:
    """
    Fit diabetes regression using both GD and analytical solution and compare.

    Return dict with:
      - "theta_l2_diff"
      - "train_mse_diff"
      - "test_mse_diff"
      - "train_r2_diff"
      - "test_r2_diff"
      - "theta_cosine_sim"

    (Cosine similarity near 1 means parameters align.)
    """
    # TODO: implement
    # Run Gradient Descent solution
    gd_train_mse, gd_test_mse, gd_train_r2, gd_test_r2, theta_gd = diabetes_linear_gd(
        lr=lr,
        epochs=epochs,
        test_size=test_size,
        seed=seed
    )

    # Run Analytical solution
    an_train_mse, an_test_mse, an_train_r2, an_test_r2, theta_an = diabetes_linear_analytical(
        test_size=test_size,
        seed=seed
    )

    # L2 difference between parameters
    theta_l2_diff = float(np.linalg.norm(theta_gd - theta_an))

    # Cosine similarity between parameters
    dot_product = np.dot(theta_gd, theta_an)
    norm_product = np.linalg.norm(theta_gd) * np.linalg.norm(theta_an)
    theta_cosine_sim = float(dot_product / norm_product) if norm_product != 0 else 0.0

    # Absolute differences in metrics
    train_mse_diff = float(abs(gd_train_mse - an_train_mse))
    test_mse_diff = float(abs(gd_test_mse - an_test_mse))
    train_r2_diff = float(abs(gd_train_r2 - an_train_r2))
    test_r2_diff = float(abs(gd_test_r2 - an_test_r2))

    return {
        "theta_l2_diff": theta_l2_diff,
        "train_mse_diff": train_mse_diff,
        "test_mse_diff": test_mse_diff,
        "train_r2_diff": train_r2_diff,
        "test_r2_diff": test_r2_diff,
        "theta_cosine_sim": theta_cosine_sim,
    }
