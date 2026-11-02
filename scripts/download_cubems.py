"""Download the public CU-BEMS dataset from Kaggle into data/raw.

Credentials are read from the environment (never commit them):
    KAGGLE_USERNAME, KAGGLE_KEY   -> from https://www.kaggle.com/settings  ("Create New Token")
    KAGGLE_DATASET                -> the dataset slug, e.g. "<owner>/<dataset-name>"
"""
import argparse
import glob
import os
import shutil
import sys
import tempfile

import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default=os.environ.get("KAGGLE_DATASET"),
                   help="Kaggle slug owner/name (or set KAGGLE_DATASET)")
    p.add_argument("--out", default="data/raw")
    args = p.parse_args()

    if not args.dataset:
        sys.exit("No dataset slug: pass --dataset or set KAGGLE_DATASET.")
    if not (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")) \
            and not os.path.exists(os.path.expanduser("~/.kaggle/kaggle.json")):
        sys.exit("No Kaggle credentials: set KAGGLE_USERNAME and KAGGLE_KEY.")

    # Imported here because the kaggle package authenticates on import.
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    os.makedirs(args.out, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        print(f"Downloading {args.dataset} ...")
        api.dataset_download_files(args.dataset, path=tmp, unzip=True, quiet=False)

        kept = 0
        for path in sorted(glob.glob(os.path.join(tmp, "**", "*.csv"), recursive=True)):
            # Keep only CU-BEMS floor files: they must have a 'Date' column and kW columns.
            header = pd.read_csv(path, nrows=0).columns
            if "Date" not in header or not any("kW" in c for c in header):
                print(f"  skip {os.path.basename(path)} (not a CU-BEMS floor file)")
                continue
            shutil.move(path, os.path.join(args.out, os.path.basename(path)))
            kept += 1
            print(f"  kept {os.path.basename(path)}")

    if kept == 0:
        sys.exit("No CU-BEMS floor CSVs found in the download. Check the slug.")
    print(f"Done: {kept} files in {args.out}")


if __name__ == "__main__":
    main()
