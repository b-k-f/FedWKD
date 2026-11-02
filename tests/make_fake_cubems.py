"""Write a tiny CU-BEMS-shaped floor file (1-min data, the 17 kW columns) for CI smoke tests."""
import argparse
import os

import numpy as np
import pandas as pd

COLUMNS = ['z1_AC1(kW)', 'z1_AC2(kW)', 'z1_AC3(kW)', 'z1_AC4(kW)', 'z1_Light(kW)', 'z1_Plug(kW)',
           'z2_AC1(kW)', 'z2_Light(kW)', 'z2_Plug(kW)', 'z3_Light(kW)', 'z3_Plug(kW)',
           'z4_AC1(kW)', 'z4_Light(kW)', 'z4_Plug(kW)', 'z5_AC1(kW)', 'z5_Light(kW)', 'z5_Plug(kW)']


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/raw")
    p.add_argument("--name", default="2018Floor3.csv")
    p.add_argument("--days", type=int, default=6)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(0)
    idx = pd.date_range("2018-03-01", periods=args.days * 1440, freq="1min")
    hour = idx.hour.values + idx.minute.values / 60
    office = ((hour > 8) & (hour < 18)).astype(float)
    df = pd.DataFrame({"Date": idx.strftime("%Y-%m-%d %H:%M:%S")})
    for c in COLUMNS:
        base = 15 if "AC" in c else 4 if "Light" in c else 3
        df[c] = (base * office + rng.normal(0, 0.5, len(idx))).clip(0)
    df["z1_S1(degC)"] = 25 + rng.normal(0, .5, len(idx))
    df.to_csv(os.path.join(args.out, args.name), index=False)
    print(f"wrote {args.name} ({args.days} days) to {args.out}")


if __name__ == "__main__":
    main()
