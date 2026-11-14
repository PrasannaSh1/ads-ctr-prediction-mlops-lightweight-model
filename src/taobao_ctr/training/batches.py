"""Turns rows of the dataset into batches of tensors for the models.

The batches are built in a background thread, so the CPU prepares the next batch while
the GPU is still training on the current one.
"""
import queue
import threading

import numpy as np
import torch

from taobao_ctr.data.history import get_history
from taobao_ctr.data.impressions import ID_COLS


def make_batch(data, rows):
    """One batch: a dict of tensors for the given rows."""
    batch = {}
    for col in ID_COLS:
        batch[col] = torch.from_numpy(data.X[col][rows].astype(np.int64))
    hist_cate, hist_brand, hist_btag = get_history(data, rows)
    batch["hist_cate"] = torch.from_numpy(hist_cate.astype(np.int64))
    batch["hist_brand"] = torch.from_numpy(hist_brand.astype(np.int64))
    batch["hist_btag"] = torch.from_numpy(hist_btag.astype(np.int64))
    batch["te"] = torch.from_numpy(data.TE[rows])
    batch["label"] = torch.from_numpy(data.y[rows])
    return batch


def fill_queue(data, rows, batch_size, batch_queue, stop_event):
    """Runs in the background thread: builds batches and puts them in the queue."""
    for start in range(0, len(rows), batch_size):
        if stop_event.is_set():
            return
        batch = make_batch(data, rows[start:start + batch_size])
        batch_queue.put(batch)
    batch_queue.put(None)  # None means "no more batches"


def batches(data, rows, batch_size, device, shuffle=False):
    """Yields batches on the device, one after another."""
    if shuffle:
        rows = np.random.permutation(rows)
    batch_queue = queue.Queue(maxsize=6)  # at most 6 batches wait in memory
    stop_event = threading.Event()
    worker = threading.Thread(target=fill_queue, args=(data, rows, batch_size, batch_queue, stop_event), daemon=True)
    worker.start()
    try:
        while True:
            batch = batch_queue.get()
            if batch is None:
                break
            batch_on_device = {}
            for key, value in batch.items():
                batch_on_device[key] = value.to(device, non_blocking=True)
            yield batch_on_device
    finally:
        # If training stopped early, tell the worker to stop and empty the queue
        stop_event.set()
        while not batch_queue.empty():
            batch_queue.get_nowait()
