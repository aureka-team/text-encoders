import os
import h5py

import numpy as np

from typing import Optional


def save_h5(save_path: str, cache_key: str, vectors: np.ndarray):
    out_file = f"{save_path}/{cache_key}.h5"
    with h5py.File(out_file, "w") as h5f:
        h5f.create_dataset("vectors", data=vectors)


def load_h5(load_path: str, cache_key: str) -> Optional[np.ndarray]:
    in_file = f"{load_path}/{cache_key}.h5"
    if not os.path.exists(in_file):
        return

    with h5py.File(in_file, "r") as h5f:
        vectors = np.array(h5f.get("vectors"))
        return vectors
