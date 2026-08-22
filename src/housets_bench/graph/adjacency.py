from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from housets_bench.graph.geo_knn import GeoGraph, _normalize_zip


def load_adjacency_graph(path: str | Path, zipcodes: Iterable[str]) -> GeoGraph:
    """Load a directly-supplied adjacency matrix and align it to ``zipcodes`` order.

    Supported formats:
      - ``.npy``: a raw ``[N, N]`` array, assumed to already be in the same
        node order as ``zipcodes``.
      - ``.csv``: a square matrix with a header row and first column of node
        ids. If those ids look like the dataset's ids (after normalization),
        the matrix is reindexed to match ``zipcodes`` order; otherwise it is
        used positionally.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"adjacency matrix not found: {p}")

    zips = [_normalize_zip(z) for z in zipcodes]
    n = len(zips)

    suffix = p.suffix.lower()
    if suffix == ".npy":
        A = np.load(p)
    elif suffix == ".csv":
        df = pd.read_csv(p, index_col=0)
        row_ids = [_normalize_zip(i) for i in df.index]
        col_ids = [_normalize_zip(c) for c in df.columns]
        if set(row_ids) == set(zips) and set(col_ids) == set(zips):
            df.index = row_ids
            df.columns = col_ids
            df = df.reindex(index=zips, columns=zips)
        A = df.to_numpy(dtype=np.float64)
    else:
        raise ValueError(f"Unsupported adjacency matrix format: {suffix} (use .npy or .csv)")

    A = np.asarray(A, dtype=np.float64)
    if A.shape != (n, n):
        raise ValueError(
            f"adjacency matrix shape {A.shape} does not match number of nodes ({n}, {n}); "
            f"the matrix must be [N,N] with N == number of ZIPs in the (possibly subsampled) dataset"
        )

    src, dst = np.nonzero(A)
    edge_weight = A[src, dst].astype(np.float32)
    edge_index = np.stack([src.astype(np.int64), dst.astype(np.int64)], axis=0)

    return GeoGraph(edge_index=edge_index, edge_weight=edge_weight)
