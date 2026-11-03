"""Run the merged Stage 1 -> Stage 2 notebook from the repository root.

Usage:
    python run_notebook.py
    python run_notebook.py --timeout 1800

The source notebook is never overwritten. Executed results are written to
notebooks/ket_qua/Project_Team_9_Tong_Stage1_Stage2_executed.ipynb.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "Project_Team_9" / "Project_Team_9_Tong_Stage1_Stage2.ipynb"
OUTPUT = ROOT / "notebooks" / "ket_qua" / "Project_Team_9_Tong_Stage1_Stage2_executed.ipynb"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=int, default=1800, help="Seconds per cell")
    args = parser.parse_args()

    if not SOURCE.is_file():
        raise FileNotFoundError(f"Missing merged notebook: {SOURCE}")
    if not (ROOT / "data_goc" / "raw_data.zip").is_file():
        raise FileNotFoundError("Missing original data: data_goc/raw_data.zip")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    # Use a headless plotting backend for noninteractive/Docker execution.
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

    notebook = nbformat.read(SOURCE, as_version=4)
    nbformat.validate(notebook)
    print(f"Executing {SOURCE.relative_to(ROOT)}")
    print(f"Code cells: {sum(c.cell_type == 'code' for c in notebook.cells)}", flush=True)
    # The notebook itself deliberately switches CWD to ROOT/Data in cell 6.
    # nbclient's initial CWD must therefore be the repository root.
    client = NotebookClient(
        notebook,
        timeout=args.timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
        allow_errors=False,
        record_timing=True,
    )
    try:
        client.execute()
    finally:
        # Keep completed cells and the failing cell's traceback on failure.
        nbformat.write(notebook, OUTPUT)
        print(f"Execution report: {OUTPUT.relative_to(ROOT)}", flush=True)
    expected = [
        ROOT / "Data/processed/data_khach_hang.csv",
        ROOT / "Data/processed/data_regression.csv",
        ROOT / "Data/processed/pca_features.csv",
        ROOT / "models/PCA.joblib",
        ROOT / "models/Classification.joblib",
        ROOT / "app/kmeans_pca.pkl",
        ROOT / "app/comparison.csv",
        ROOT / "notebooks/ket_qua/so_sanh_mo_hinh.csv",
    ]
    missing = [str(p.relative_to(ROOT)) for p in expected if not p.is_file()]
    if missing:
        raise RuntimeError(f"Notebook ended but output files are missing: {missing}")
    print("SUCCESS: all notebook code cells executed and expected outputs exist.", flush=True)


if __name__ == "__main__":
    main()
