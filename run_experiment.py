"""Run FedWKD or one of the baselines on the client CSVs, using the repo's own modules.

    python run_experiment.py --algo fedwkd --data-dir data/clients --n-clients 100 --rounds 20

Algorithms: fedwkd (proposed), fedavg, cadis, fedgkd.
Outputs in results/<algo>/: metrics.csv, summary.json, convergence.png, forecast.png
"""
import argparse
import json
import os
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.preprocessing import MinMaxScaler  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

import algorithm as A  # noqa: E402
from data import fetch_dataset  # noqa: E402
from model import LSTMModel, device  # noqa: E402
from util import move_sliding_window  # noqa: E402

FEATURES = ["total_demand", "AC", "Light", "Plug", "Year", "Month", "Day", "Hour", "Minute"]
LABEL_COL = 0  # forecast total_demand
MODEL_TYPE = "LSTM"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loaders(dfs, window, batch_size, test_frac):
    """Per-client chronological split; scalers are fitted on the train part only."""
    train_loaders, test_loaders, label_scalers, names = [], [], [], []
    for name in sorted(dfs):
        values = dfs[name][FEATURES].values.astype(np.float64)
        split = int(len(values) * (1 - test_frac))
        if split <= window + batch_size or len(values) - split <= window:
            print(f"skip {name}: too short ({len(values)} rows)")
            continue

        x_sc = MinMaxScaler().fit(values[:split])
        y_sc = MinMaxScaler().fit(values[:split, [LABEL_COL]])
        scaled = x_sc.transform(values)
        scaled[:, LABEL_COL] = y_sc.transform(values[:, [LABEL_COL]]).ravel()

        X, y = move_sliding_window(scaled, window, list(range(len(FEATURES))), LABEL_COL)
        cut = split - window
        to_t = lambda a: torch.tensor(a, dtype=torch.float32)  # noqa: E731
        train_ds = TensorDataset(to_t(X[:cut]), to_t(y[:cut]))
        test_ds = TensorDataset(to_t(X[cut:]), to_t(y[cut:]))

        # drop_last: the models use a fixed-size hidden state of batch_size
        train_loaders.append(DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True))
        test_loaders.append(DataLoader(test_ds, batch_size=batch_size, shuffle=False))
        label_scalers.append(y_sc)
        names.append(name)
    return train_loaders, test_loaders, label_scalers, names


def run(args, train_loaders, test_loaders, label_sc):
    n = len(train_loaders)
    model = LSTMModel(len(FEATURES), args.hidden, 1, args.layers).to(device)
    common = dict(n_clients=n, batch_size=args.batch_size, num_local_epochs=args.local_epochs,
                  lr=args.lr, model_type=MODEL_TYPE, device=device)

    if args.algo == "fedwkd":
        # Round 0: FedAvg warm-up to obtain an initial global model and global logits,
        # then weighted aggregation + KD for the remaining rounds (algorithm.fedkd).
        _, _, l0, s0, m0, r0, global_model, agg_logits = A.fedavg(
            model, train_loaders, test_loaders, label_sc, max_rounds=1, **common)
        out, tgt, loss, smape, mae, rmse = A.fedkd(
            global_model, agg_logits, train_loaders, test_loaders, label_sc,
            max_rounds=args.rounds, **common)
        return out, tgt, l0 + loss, s0 + smape, m0 + mae, r0 + rmse
    if args.algo == "fedavg":
        out, tgt, loss, smape, mae, rmse, _, _ = A.fedavg(
            model, train_loaders, test_loaders, label_sc, max_rounds=args.rounds, **common)
        return out, tgt, loss, smape, mae, rmse
    if args.algo == "cadis":
        return A.cadis(model, train_loaders, test_loaders, args.lambda_kd, label_sc,
                       max_rounds=args.rounds, **common)
    if args.algo == "fedgkd":
        return A.fedgkd(model, train_loaders, test_loaders, label_sc, max_rounds=args.rounds,
                        buffer_size=args.buffer_size, kd_weight=args.kd_weight, **common)
    raise ValueError(args.algo)


def save_outputs(args, outputs, targets, loss, smape, mae, rmse, elapsed, names):
    out_dir = os.path.join(args.out, args.algo)
    os.makedirs(out_dir, exist_ok=True)

    metrics = pd.DataFrame({"round": range(1, len(loss) + 1), "loss": loss,
                            "smape": smape, "mae": mae, "rmse": rmse})
    metrics.to_csv(os.path.join(out_dir, "metrics.csv"), index=False)

    summary = {
        "algo": args.algo, "clients": len(names), "rounds": len(loss),
        "final": metrics.iloc[-1].drop("round").astype(float).round(6).to_dict(),
        "best_rmse": float(metrics["rmse"].min()),
        "wall_time_sec": round(elapsed, 1), "device": str(device),
        "config": {k: v for k, v in vars(args).items() if k not in ("out",)},
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    for ax, col in zip(axes, ["loss", "rmse", "smape"]):
        ax.plot(metrics["round"], metrics[col], marker="o", ms=3)
        ax.set_title(col.upper()); ax.set_xlabel("round"); ax.grid(alpha=.3)
    fig.suptitle(f"{args.algo} - {len(names)} clients"); fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "convergence.png"), dpi=120); plt.close(fig)

    if outputs:
        k = min(len(outputs[0]), 500)
        fig, ax = plt.subplots(figsize=(12, 3.5))
        ax.plot(targets[0][:k], label="actual", lw=1)
        ax.plot(outputs[0][:k], label="forecast", lw=1)
        ax.set_title(f"{args.algo} - total demand forecast, first test client")
        ax.set_ylabel("kW"); ax.legend(); fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "forecast.png"), dpi=120); plt.close(fig)

    print(json.dumps(summary, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", default="fedwkd", choices=["fedwkd", "fedavg", "cadis", "fedgkd"])
    p.add_argument("--data-dir", default="data/clients")
    p.add_argument("--out", default="results")
    p.add_argument("--n-clients", type=int, default=100)
    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("--local-epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--window", type=int, default=6, help="input steps (6 x 10 min = 1 h)")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--lambda-kd", type=float, default=1.0, help="CADIS")
    p.add_argument("--buffer-size", type=int, default=5, help="FedGKD")
    p.add_argument("--kd-weight", type=float, default=0.5, help="FedGKD")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    dfs = fetch_dataset(args.data_dir)
    dfs = {k: dfs[k] for k in sorted(dfs)[:args.n_clients]}
    train_l, test_l, label_sc, names = build_loaders(dfs, args.window, args.batch_size, args.test_frac)
    if len(train_l) < 2:
        raise SystemExit("Need at least 2 usable clients.")
    print(f"{len(names)} clients on {device}")

    t0 = time.time()
    outputs, targets, loss, smape, mae, rmse = run(args, train_l, test_l, label_sc)
    save_outputs(args, outputs, targets, loss, smape, mae, rmse, time.time() - t0, names)


if __name__ == "__main__":
    main()
