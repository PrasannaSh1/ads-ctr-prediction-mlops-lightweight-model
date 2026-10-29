"""make_batch() and the background-thread batches() generator that overlaps history gathering
(CPU) with model training (GPU) (notebook section 5)."""
from __future__ import annotations

import queue
import threading

import numpy as np
import torch

from taobao_ctr.data.dataset import Dataset
from taobao_ctr.data.history import history
from taobao_ctr.data.impressions import ID_COLS


def make_batch(data: Dataset, rows: np.ndarray) -> dict:
    b = {c: torch.from_numpy(data.X[c][rows].astype(np.int64)) for c in ID_COLS}
    hc, hb, ht = history(rows, first=data.first, end=data.end, BC=data.BC, BB=data.BB, BT=data.BT, seq_len=data.seq_len)
    b["hist_cate"], b["hist_brand"], b["hist_btag"] = (torch.from_numpy(h.astype(np.int64)) for h in (hc, hb, ht))
    b["te"] = torch.from_numpy(data.TE[rows])
    b["label"] = torch.from_numpy(data.y[rows])
    return b


def batches(data: Dataset, rows: np.ndarray, batch_size: int, device: torch.device, shuffle: bool = False):
    """Yields batches on `device`, built in a background thread."""
    if shuffle:
        rows = np.random.permutation(rows)
    q, stop = queue.Queue(maxsize=6), threading.Event()

    def producer():
        for s in range(0, len(rows), batch_size):
            if stop.is_set():
                return
            q.put(make_batch(data, rows[s:s + batch_size]))
        q.put(None)

    threading.Thread(target=producer, daemon=True).start()
    try:
        while (b := q.get()) is not None:
            yield {k: v.to(device, non_blocking=True) for k, v in b.items()}
    finally:  # early stop: let the producer finish
        stop.set()
        while not q.empty():
            q.get_nowait()
