"""
PCAM Precision Agent — Ablation Study
======================================

Systematically varies design parameters to demonstrate the principled
nature of each design choice, and generates a comparison table.

Tests:
  1. Baseline (Pi=I)
  2. Our full agent
  3. Without twin disambiguation
  4. Different k-NN values (k=1, k=3, k=5, k=10)
  5. Different reliability ranges
  6. Random precision (sanity check)
  7. Inverse reliability (ablation — should fail)

Usage:
    python ablation_study.py

Author: Team Antigravity
"""

from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pcam_model import PCAMModel, build_default_R
from data import make_patterns, make_test_queries
from checks import retrieval_accuracy


class BaselineAgent:
    """Pi = I everywhere (the floor)."""
    def __init__(self, X, params):
        self.N = X.shape[1]
    def predict_precision(self, q):
        return np.ones(self.N)


class RandomAgent:
    """Random precision (sanity check — should be near or below baseline)."""
    def __init__(self, X, params):
        self.N = X.shape[1]
        self.rng = np.random.default_rng(12345)
    def predict_precision(self, q):
        return self.rng.exponential(size=self.N)


class InverseAgent:
    """Inverse of our approach — up-weight noisy dims, down-weight clean ones."""
    def __init__(self, X, params):
        self.X = np.asarray(X, dtype=np.float64)
        self.K, self.N = self.X.shape
        self.dim_std = self.X.std(axis=0) + 1e-8
        norms = np.linalg.norm(self.X, axis=1, keepdims=True)
        self.X_normed = self.X / np.maximum(norms, 1e-12)

    def predict_precision(self, q):
        q = np.asarray(q, dtype=np.float64)
        q_norm = np.linalg.norm(q)
        if q_norm < 1e-12:
            return np.ones(self.N)
        q_unit = q / q_norm
        cosines = self.X_normed @ q_unit
        best = np.argmax(cosines)
        residual = np.abs(q - self.X[best]) / self.dim_std
        # INVERSE: up-weight noisy dims
        raw = 1.0 - np.exp(-residual)
        return 0.3 + 2.7 * raw


class VariantAgent:
    """Parameterised version of our agent for ablation."""
    def __init__(self, X, params, k=5, beta_soft=5.0, pi_low=0.3, pi_high=3.0,
                 twin_enabled=True, twin_threshold=0.15, twin_weight=0.3):
        self.X = np.asarray(X, dtype=np.float64)
        self.K, self.N = self.X.shape
        self.dim_std = self.X.std(axis=0) + 1e-8
        norms = np.linalg.norm(self.X, axis=1, keepdims=True)
        self.X_normed = self.X / np.maximum(norms, 1e-12)
        self.k = k
        self.beta_soft = beta_soft
        self.pi_low = pi_low
        self.pi_high = pi_high
        self.twin_enabled = twin_enabled
        self.twin_threshold = twin_threshold
        self.twin_weight = twin_weight

    def predict_precision(self, q):
        q = np.asarray(q, dtype=np.float64)
        q_norm = np.linalg.norm(q)
        if q_norm < 1e-12:
            return np.ones(self.N)
        q_unit = q / q_norm
        cosines = self.X_normed @ q_unit
        k = min(self.k, self.K)
        if k >= self.K:
            top_k = np.arange(self.K)
        else:
            top_k = np.argpartition(-cosines, k)[:k]
        cos_top = cosines[top_k]
        cos_shift = cos_top - cos_top.max()
        weights = np.exp(self.beta_soft * cos_shift)
        weights /= weights.sum() + 1e-10
        centroid = weights @ self.X[top_k]
        residual = np.abs(q - centroid) / self.dim_std
        raw = np.exp(-residual)
        precision = self.pi_low + (self.pi_high - self.pi_low) * raw

        if self.twin_enabled and len(top_k) >= 2:
            sorted_idx = np.argsort(-cos_top)
            i1, i2 = top_k[sorted_idx[0]], top_k[sorted_idx[1]]
            gap = cosines[i1] - cosines[i2]
            if gap < self.twin_threshold:
                diff = np.abs(self.X[i1] - self.X[i2])
                diff_norm = diff / (diff.max() + 1e-8)
                dw = self.twin_weight * (1.0 - gap / self.twin_threshold)
                precision = precision * (1.0 + dw * diff_norm)

        if not np.all(np.isfinite(precision)):
            return np.ones(self.N)
        return precision


def run_ablation(seed=42):
    K, N = 16, 64
    X = make_patterns(K=K, N=N, seed=seed)
    R = build_default_R(N=N, seed=seed)
    model = PCAMModel(X, R)
    params = {"R": R, "eta": model.eta, "beta": model.beta,
              "dt": model.dt, "T_max": model.T_max, "tol": model.tol,
              "T_in": model.T_in, "pi_min": model.pi_min, "pi_max": model.pi_max}

    queries, truths, _ = make_test_queries(X, [0.5, 0.7, 0.8], 100, seed=seed)

    experiments = [
        ("Baseline (Pi=I)", BaselineAgent(X, params)),
        ("Full Agent (ours)", VariantAgent(X, params)),
        ("No Twin Disambig", VariantAgent(X, params, twin_enabled=False)),
        ("k=1 (hard NN)", VariantAgent(X, params, k=1)),
        ("k=3", VariantAgent(X, params, k=3)),
        ("k=5 (default)", VariantAgent(X, params, k=5)),
        ("k=10", VariantAgent(X, params, k=10)),
        ("Tight range [0.5,2.0]", VariantAgent(X, params, pi_low=0.5, pi_high=2.0)),
        ("Wide range [0.1,5.0]", VariantAgent(X, params, pi_low=0.1, pi_high=5.0)),
        ("beta_soft=1.0", VariantAgent(X, params, beta_soft=1.0)),
        ("beta_soft=10.0", VariantAgent(X, params, beta_soft=10.0)),
        ("Random Pi", RandomAgent(X, params)),
        ("Inverse reliability", InverseAgent(X, params)),
    ]

    print(f"PCAM Ablation Study (seed={seed})")
    print(f"{'='*70}")
    print(f"{'Experiment':<25} {'Accuracy':>10} {'Delta':>10} {'Status':>10}")
    print(f"{'-'*70}")

    baseline_acc = None
    for name, agent in experiments:
        t0 = time.time()
        acc = retrieval_accuracy(model, agent, queries, truths)
        dt = time.time() - t0
        if baseline_acc is None:
            baseline_acc = acc
            delta = 0.0
        else:
            delta = acc - baseline_acc

        status = "✓ PASS" if delta > 0 else ("— BASE" if delta == 0 else "✗ FAIL")
        print(f"  {name:<25} {acc:>8.4f}   {delta:>+8.4f}   {status}")

    print(f"{'='*70}")


if __name__ == "__main__":
    seeds = [42, 101]
    for s in seeds:
        run_ablation(s)
        print()
