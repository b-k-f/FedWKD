"""Collect results/<algo>/ into a comparison table and plot (also written to the GitHub job summary)."""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    args = p.parse_args()

    rows, curves = [], {}
    for path in sorted(glob.glob(os.path.join(args.results, "*", "summary.json"))):
        s = json.load(open(path))
        rows.append({"algorithm": s["algo"], "clients": s["clients"], "rounds": s["rounds"],
                     "final RMSE": s["final"]["rmse"], "final MAE": s["final"]["mae"],
                     "final sMAPE %": s["final"]["smape"], "best RMSE": s["best_rmse"],
                     "time (s)": s["wall_time_sec"]})
        curves[s["algo"]] = pd.read_csv(os.path.join(os.path.dirname(path), "metrics.csv"))
    if not rows:
        raise SystemExit("No results found.")

    table = pd.DataFrame(rows).sort_values("final RMSE")
    md = "## FedWKD vs baselines\n\n" + table.to_markdown(index=False, floatfmt=".4f") + "\n"
    with open(os.path.join(args.results, "comparison.md"), "w") as f:
        f.write(md)
    print(md)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write(md)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, col in zip(axes, ["rmse", "mae", "smape"]):
        for algo, m in curves.items():
            ax.plot(m["round"], m[col], label=algo, lw=1.5)
        ax.set_title(col.upper()); ax.set_xlabel("round"); ax.grid(alpha=.3)
    axes[0].legend(); fig.tight_layout()
    fig.savefig(os.path.join(args.results, "comparison.png"), dpi=130)


if __name__ == "__main__":
    main()
