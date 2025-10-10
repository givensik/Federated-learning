# /shared/data/prepare_sensor_clients_5.py
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import MinMaxScaler

DATA_DIR = Path(__file__).resolve().parent
RAW_DIR = DATA_DIR / "raw"
OUT_DIR = DATA_DIR / "processed"
OUT_DIR.mkdir(exist_ok=True, parents=True)

CLIENT_BATCH_MAP = {
    1: [1, 2],
    2: [3, 4],
    3: [5, 6],
    4: [7, 8],
    5: [9, 10]
}

def load_svmlib(path):
    X, y = load_svmlight_file(str(path), n_features=128, dtype=np.float64)
    return X.toarray(), y.astype(int)

def make_client_csv(cid, batch_list):
    X_all, y_all = [], []
    for b in batch_list:
        f = RAW_DIR / f"batch{b}.dat"
        if not f.exists():
            print(f"⚠️ batch{b}.dat not found, skipping")
            continue
        X, y = load_svmlib(f)
        X_all.append(X)
        y_all.append(y)
    if not X_all:
        return
    X = np.vstack(X_all)
    y = np.concatenate(y_all)
    print(f"[Client{cid}] total samples: {len(y)} from batches {batch_list}")
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_scaled = scaler.fit_transform(X)
    df = pd.DataFrame(X_scaled, columns=[f"f{i+1}" for i in range(X.shape[1])])
    df["label"] = y
    df.to_csv(OUT_DIR / f"client{cid}.csv", index=False)
    print(f"✅ client{cid}.csv saved ({len(df)} rows)")

def main():
    print(DATA_DIR)
    for cid, batches in CLIENT_BATCH_MAP.items():
        make_client_csv(cid, batches)

if __name__ == "__main__":
    main()
