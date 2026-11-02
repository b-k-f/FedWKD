"""Generate the synthetic CU-BEMS clients with DoppelGANger (DGAN).

Port of the thesis notebook gan-18col-2018.ipynb (repo b-k-f/EIF-FL):
  * one CU-BEMS floor file (default 2018Floor3.csv), the 17 zone-level kW columns
  * rows with all 17 values missing are dropped, the 2018-10-11 -> 2018-11-21 outage is removed,
    and the series is re-indexed as continuous 1-minute data ending on 31 December
  * the series is cut into 1-day windows (1440 x 17) to train the DGAN
  * DGANConfig identical to the notebook (lr 1e-4, sample_len 1440, 10 000 epochs)
  * N clients are generated and written as <year>Floor<i>.csv (Date + 17 kW columns, 1-min),
    i.e. the raw CU-BEMS format that FedKWD's data.py reads and resamples to 10 minutes.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

COLUMNS = ['z1_AC1(kW)', 'z1_AC2(kW)', 'z1_AC3(kW)', 'z1_AC4(kW)', 'z1_Light(kW)', 'z1_Plug(kW)',
           'z2_AC1(kW)', 'z2_Light(kW)', 'z2_Plug(kW)', 'z3_Light(kW)', 'z3_Plug(kW)',
           'z4_AC1(kW)', 'z4_Light(kW)', 'z4_Plug(kW)', 'z5_AC1(kW)', 'z5_Light(kW)', 'z5_Plug(kW)']
WINDOW = 1440  # one day of 1-minute observations


def load_dgan():
    """Use the vendored gan/ package from the thesis if it is in the repo, else gretel-synthetics."""
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    try:
        from gan.dgan import DGAN
        from gan.config import DGANConfig, OutputType
        source = "vendored gan/ package"
    except ImportError:
        from gretel_synthetics.timeseries_dgan.dgan import DGAN
        from gretel_synthetics.timeseries_dgan.config import DGANConfig, OutputType
        source = "gretel-synthetics"
    print(f"DGAN implementation: {source}")
    return DGAN, DGANConfig, OutputType


def energy_dataset(filename, year, gap=None):
    """Same steps as EnergyDataset() in the notebook."""
    df = pd.read_csv(filename)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df[COLUMNS] = df[COLUMNS].replace('', np.nan)
    df = df.dropna(subset=COLUMNS, how='all')
    df = df.sort_values('Date').rename(columns={'Date': 'datetime'})

    if gap:
        start, end = pd.Timestamp(gap[0]), pd.Timestamp(gap[1])
        df = df[(df['datetime'] < start) | (df['datetime'] > end)]

    # Re-index as continuous 1-minute data ending at the end of the year (as in the notebook)
    df.index = pd.date_range(end=pd.Timestamp(f'{year}-12-31 23:59:00'), freq='1min', periods=len(df))
    df = df.drop(columns='datetime')[COLUMNS]
    return df.dropna(subset=COLUMNS, how='all')


def make_config(DGANConfig, n_windows, epochs):
    return DGANConfig(
        max_sequence_len=WINDOW,
        sample_len=WINDOW,
        batch_size=min(1000, n_windows),
        apply_feature_scaling=True,
        apply_example_scaling=False,
        use_attribute_discriminator=False,
        generator_learning_rate=0.0001,
        discriminator_learning_rate=0.0001,
        epochs=epochs,
    )


def client_diversity(clients):
    """Mean absolute difference between consecutive clients, relative to the mean load."""
    diffs = [np.mean(np.abs(a - b)) for a, b in zip(clients[:-1], clients[1:])]
    scale = np.mean([np.mean(np.abs(c)) for c in clients]) or 1.0
    return float(np.mean(diffs) / scale)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", default="data/raw")
    p.add_argument("--floor-file", default="2018Floor3.csv")
    p.add_argument("--year", type=int, default=None, help="default: first 4 characters of the file name")
    p.add_argument("--gap", nargs=2, default=None, metavar=("START", "END"),
                   help="outage to remove; default for 2018: 2018-10-11 16:11 2018-11-21 11:52")
    p.add_argument("--out-dir", default="data/clients")
    p.add_argument("--n-clients", type=int, default=100)
    p.add_argument("--epochs", type=int, default=10000)
    p.add_argument("--model-path", default="saved_models/dgan.pt",
                   help="loaded if it exists (skips training), otherwise saved after training")
    p.add_argument("--retrain", action="store_true", help="train even if --model-path exists")
    p.add_argument("--noise", choices=["notebook", "independent"], default="notebook",
                   help="notebook: same call as the thesis notebook; independent: fresh noise per client")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    import torch
    DGAN, DGANConfig, OutputType = load_dgan()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    year = args.year or int(args.floor_file[:4])
    gap = args.gap or (["2018-10-11 16:11", "2018-11-21 11:52"] if year == 2018 else None)
    sensor_df = energy_dataset(os.path.join(args.raw_dir, args.floor_file), year, gap)

    features = sensor_df.to_numpy(dtype=np.float64)
    n = features.shape[0] // WINDOW
    if n < 2:
        sys.exit(f"Only {n} full day(s) of data in {args.floor_file}; DGAN needs more.")
    features = features[:n * WINDOW].reshape(-1, WINDOW, features.shape[1])
    print(f"training windows: {features.shape}")  # (#days, 1440, 17)

    if os.path.exists(args.model_path) and not args.retrain:
        print(f"Loading trained DGAN from {args.model_path}")
        model = DGAN.load(args.model_path, map_location=torch.device("cpu")
                          if not torch.cuda.is_available() else None)
    else:
        model = DGAN(make_config(DGANConfig, features.shape[0], args.epochs))
        print(f"Training DGAN for {args.epochs} epochs on {'GPU' if torch.cuda.is_available() else 'CPU'}")
        model.train_numpy(features, feature_types=[OutputType.CONTINUOUS] * features.shape[2])
        os.makedirs(os.path.dirname(args.model_path) or ".", exist_ok=True)
        model.save(args.model_path)
        print(f"Saved DGAN to {args.model_path}")

    os.makedirs(args.out_dir, exist_ok=True)
    feature_noise = torch.randn(features.shape[0], WINDOW, 1)
    generated, manifest = [], []
    for i in range(args.n_clients):
        if args.noise == "notebook":
            # Exactly as in the notebook loop (rate is reset to 0.0001 at the top of each iteration)
            rate = 0.0001
            _, synthetic = model.generate_numpy(features.shape[0], feature_noise=feature_noise * rate)
        else:
            torch.manual_seed(args.seed + i + 1)
            _, synthetic = model.generate_numpy(features.shape[0])

        synthetic = np.asarray(synthetic).reshape(-1, len(COLUMNS))
        out = pd.DataFrame(synthetic, columns=COLUMNS)
        out.insert(0, "Date", sensor_df.index[len(sensor_df) - len(out):].strftime("%Y-%m-%d %H:%M:%S"))
        fname = f"{year}Floor{i}.csv"
        out.to_csv(os.path.join(args.out_dir, fname), index=False)
        generated.append(synthetic)
        manifest.append({"client": fname, "rows": len(out)})
        print(f"client {i + 1}/{args.n_clients}: {fname}")

    diversity = client_diversity(generated) if len(generated) > 1 else None
    if diversity is not None:
        print(f"client diversity (mean |client_i - client_i+1| / mean load): {diversity:.6f}")
        if diversity < 1e-3:
            print("WARNING: the generated clients are (almost) identical. "
                  "Use --noise independent for distinct clients.")

    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump({"floor_file": args.floor_file, "epochs": args.epochs, "noise": args.noise,
                   "seed": args.seed, "diversity": diversity, "clients": manifest}, f, indent=2)


if __name__ == "__main__":
    main()
