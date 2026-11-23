"""Transfer check on a second industry: Criteo Uplift v2.1 (online advertising).

13.98M users, randomised ad exposure (85% treated, 15% control), 12 anonymised numeric
features, outcome `visit`. If the same pipeline (logistic T-learner vs response model, same
metrics) gives the same qualitative answer here, the method is not tied to one retailer.

Data: Diemert et al., "A Large Scale Benchmark for Uplift Modeling", AdKDD 2018 (CC BY-NC-SA 4.0).
"""
from __future__ import annotations

import pandas as pd

from .data import PROCESSED, RAW, connect
from .models import FeatureSpec

CRITEO_PARQUET = PROCESSED / "criteo.parquet"
CRITEO_SPEC = FeatureSpec(num=tuple(f"f{i}" for i in range(12)))


def load_criteo_sample(n: int = 3_000_000, seed: int = 0) -> pd.DataFrame:
    """Uniform random sample; columns renamed to the project convention (treatment_flg, target=visit)."""
    con = connect()
    if not CRITEO_PARQUET.exists():
        con.sql(f"COPY (SELECT * FROM read_csv_auto('{(RAW / 'criteo-uplift-v2.1.csv.gz').as_posix()}')) "
                f"TO '{CRITEO_PARQUET.as_posix()}' (FORMAT parquet)")
    con.sql(f"SELECT setseed({seed / 1000})")
    df = con.sql(f"""
        SELECT * FROM '{CRITEO_PARQUET.as_posix()}'
        USING SAMPLE reservoir({n} ROWS) REPEATABLE ({seed})
    """).df()
    return df.rename(columns={"treatment": "treatment_flg", "visit": "target"})
