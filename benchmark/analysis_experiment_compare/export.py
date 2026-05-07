from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if df is None or df.empty:
        # Write an empty csv with no rows to make pipeline outputs explicit.
        pd.DataFrame().to_csv(path, index=False)
    else:
        df.to_csv(path, index=False)


def write_comparison_set(
    out_dir: Path,
    prefix: str,
    comp: Dict[str, pd.DataFrame],
) -> None:
    write_csv(comp.get("wide", pd.DataFrame()), out_dir / f"{prefix}_wide.csv")
    write_csv(comp.get("delta", pd.DataFrame()), out_dir / f"{prefix}_delta.csv")

