"""Build the final LB=0.43961 submission ZIP from component predictions.

Reproduces the winning blend formula:
    final = 0.400 * hier_repro_s42 + 0.425 * s12_all_repro + 0.175 * film6_hier
    final = clip(final, 20, 69)

where film6_hier = median(film_s0, film_s1, film_s2, film_s3, film_s4, film_s42, hier_repro_s42)

Inputs (all in ../server_preds/):
    hier_repro_s42_test.npy     # 477 floats
    film_s{0,1,2,3,4,42}_repro_test.npy  # each 477 floats
    s12_all_repro_test.csv      # 477 rows (uuid, age) — s12 script saves CSV not npy
    test_uuids.txt              # 477 uuids in order

Output (in ../submissions/):
    v6_grid_closest_to_ref.csv  # competition CSV
    v6_grid_closest_to_ref.zip  # CodaBench upload (contains submission.csv)
"""
from __future__ import annotations

import csv
import zipfile
from pathlib import Path

import numpy as np


HERE = Path(__file__).parent
PRED_DIR = HERE.parent / "server_preds"
SUB_DIR = HERE.parent / "submissions"


def load_csv_predictions(path: Path) -> tuple[list[str], np.ndarray]:
    """Read a (uuid, age) CSV and return uuids + ages array."""
    uuids: list[str] = []
    vals: list[float] = []
    with open(path) as f:
        next(f)  # skip header
        for line in f:
            u, a = line.strip().split(",")
            uuids.append(u)
            vals.append(float(a))
    return uuids, np.array(vals, dtype=np.float64)


def main() -> None:
    # Reference uuid order
    with open(PRED_DIR / "test_uuids.txt") as f:
        ref_uuids = [line.strip() for line in f if line.strip()]

    # Component test predictions
    hier_test = np.load(PRED_DIR / "hier_repro_s42_test.npy").astype(np.float64)
    film_seeds = [
        np.load(PRED_DIR / f"film_s{s}_repro_test.npy").astype(np.float64)
        for s in [0, 1, 2, 3, 4, 42]
    ]

    # s12_all is saved as a CSV by train_sprint12.py (not numpy)
    s12_uuids, s12_test = load_csv_predictions(PRED_DIR / "s12_all_repro_test.csv")
    if s12_uuids != ref_uuids:
        # Reorder to match canonical order
        idx_map = {u: i for i, u in enumerate(s12_uuids)}
        s12_test = np.array([s12_test[idx_map[u]] for u in ref_uuids])

    # Build film6_hier (median of 6 FiLM + hier)
    film6_hier_test = np.median(np.stack(film_seeds + [hier_test]), axis=0)
    film6_hier_test = np.clip(film6_hier_test, 20, 69)

    # Final blend with grid-search-optimized weights (LB=0.43961)
    final_test = 0.400 * hier_test + 0.425 * s12_test + 0.175 * film6_hier_test
    final_test = np.clip(final_test, 20, 69)

    SUB_DIR.mkdir(exist_ok=True)
    csv_path = SUB_DIR / "v6_grid_closest_to_ref.csv"
    zip_path = SUB_DIR / "v6_grid_closest_to_ref.zip"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["uuid", "age"])
        for u, p in zip(ref_uuids, final_test):
            w.writerow([u, f"{p:.6f}"])
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(csv_path, arcname="submission.csv")

    print(f"  N predictions:  {len(final_test)}")
    print(f"  mean / std:     {final_test.mean():.3f} / {final_test.std():.3f}")
    print(f"  min / max:      {final_test.min():.2f} / {final_test.max():.2f}")
    print(f"  CSV:            {csv_path}")
    print(f"  ZIP:            {zip_path}")


if __name__ == "__main__":
    main()
