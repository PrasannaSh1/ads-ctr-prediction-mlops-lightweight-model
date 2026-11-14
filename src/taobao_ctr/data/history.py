"""Builds the "last 50 actions" history of every impression without using too much RAM.

The behaviour log has 723M rows (24 GB), so it is done in three steps:

1. stream_behaviour(): read the CSV once, in chunks, and write every row into one of
   N smaller Parquet files ("buckets"), chosen by user id.
2. build_history(): open one bucket at a time, sort it by (user, time), and keep only the
   actions that can ever appear in some impression's last-50 window. Every impression gets
   two pointers (first, end) into this compact store.
3. get_history(): for a batch of impressions, read the 50 actions before each one.

History always means actions strictly BEFORE the impression.
"""
import gc
import os
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from taobao_ctr.data.download import kaggle_csv
from taobao_ctr.data.impressions import encode

# Action types. 0 is kept for padding (an empty slot).
BTAG = {"pv": 1, "cart": 2, "fav": 3, "buy": 4}


def bucket_folder(data_dir):
    return os.path.join(data_dir, "buckets")


def stream_behaviour(imp, data_dir, kaggle_dataset, n_buckets, chunk_rows):
    """Read behavior_log.csv once and split it into n_buckets Parquet files by user.

    Returns the distinct categories and brands seen in the log.
    If this already ran before (done.npz exists), that result is loaded instead.
    """
    folder = bucket_folder(data_dir)
    done_file = os.path.join(folder, "done.npz")
    if os.path.exists(done_file):
        return np.load(done_file)
    os.makedirs(folder, exist_ok=True)
    start_time = time.time()

    # The file has some broken timestamps, so keep only a sensible time range
    t_lo = imp["ts"].min() - 30 * 86400
    t_hi = imp["ts"].max()
    user_index = pd.Index(imp["user_ids"])

    writers = [None] * n_buckets  # one Parquet writer per bucket, created when first needed
    all_cates = []
    all_brands = []
    n_read = 0

    csv_path = kaggle_csv("behavior_log.csv", data_dir, kaggle_dataset)
    reader = pd.read_csv(
        csv_path,
        chunksize=chunk_rows,
        usecols=["user", "time_stamp", "btag", "cate", "brand"],
        dtype={"user": "int32", "time_stamp": "int64", "btag": "category", "cate": "int32", "brand": "float64"},
    )
    chunk_number = 0
    for chunk in reader:
        n_read += len(chunk)

        # Drop rows with broken timestamps
        in_range = chunk["time_stamp"].between(t_lo, t_hi)
        chunk = chunk[in_range]

        # Keep only users who saw at least one ad, numbered 1..U like in the impressions
        u = user_index.get_indexer(chunk["user"].values)
        keep = u >= 0
        chunk = chunk[keep]
        u = (u[keep] + 1).astype(np.int32)

        cate = chunk["cate"].values.astype(np.int64)
        brand = chunk["brand"].fillna(-1).values.astype(np.int64)
        all_cates.append(np.unique(cate))
        all_brands.append(np.unique(brand))

        # Which bucket does each row go to? Sort the rows by bucket, so each bucket is one slice
        bucket = u % n_buckets
        order = np.argsort(bucket, kind="stable")
        btag = chunk["btag"].astype(str).map(BTAG).fillna(0).astype("int8").values
        table = pa.table({
            "u": u[order],
            "t": chunk["time_stamp"].values[order],
            "b": btag[order],
            "c": cate[order],
            "r": brand[order],
        })
        # cuts[k] = the first row of bucket k after sorting
        cuts = np.searchsorted(bucket[order], np.arange(n_buckets + 1))
        for k in range(n_buckets):
            if writers[k] is None:
                file_path = os.path.join(folder, f"b{k}.parquet")
                writers[k] = pq.ParquetWriter(file_path, table.schema)
            writers[k].write_table(table.slice(cuts[k], cuts[k + 1] - cuts[k]))

        if chunk_number % 10 == 0:
            minutes = (time.time() - start_time) / 60
            print(f"  {n_read:,} rows read | {minutes:.1f} min")
        chunk_number += 1

    for writer in writers:
        writer.close()

    cate_seen = np.unique(np.concatenate(all_cates))
    brand_seen = np.unique(np.concatenate(all_brands))
    np.savez(done_file, cate=cate_seen, brand=brand_seen)
    minutes = (time.time() - start_time) / 60
    print(f"streamed {n_read:,} behaviour rows in {minutes:.1f} min")
    return np.load(done_file)


def build_vocab(ad_cate, ad_brand, seen):
    """Sorted array of every category id and of every brand id.

    It joins the ads' own values with everything seen in the behaviour log, so the ad and the
    history use the same numbers (and later the same embeddings).
    """
    cate_vocab = np.unique(np.concatenate([ad_cate[ad_cate >= 0], seen["cate"]]))
    brand_seen = seen["brand"]
    brand_vocab = np.unique(np.concatenate([ad_brand[ad_brand >= 0], brand_seen[brand_seen >= 0]]))
    return cate_vocab, brand_vocab


def join_chunks(chunks):
    """Join a list of arrays into one array, freeing each piece as we go (saves RAM)."""
    total = 0
    for c in chunks:
        total += len(c)
    out = np.empty(total, chunks[0].dtype)
    position = 0
    while len(chunks) > 0:
        c = chunks.pop(0)
        out[position:position + len(c)] = c
        position += len(c)
    return out


def build_history(imp, cate_vocab, brand_vocab, data_dir, n_buckets, seq_len):
    """Build the compact history store and the (first, end) pointers of every impression.

    Returns first, end, BC, BB, BT:
      BC, BB, BT   = category, brand and action type of every kept action (one long array each)
      first[i]     = where impression i's user starts in those arrays
      end[i]       = just after the last action BEFORE impression i
    """
    folder = bucket_folder(data_dir)
    u_imp = imp["u"]
    ts = imp["ts"]
    n_users = len(imp["user_ids"]) + 1

    # For every user: the time of their first and of their last impression (ts is sorted)
    first_t = np.zeros(n_users, np.int64)
    last_t = np.zeros(n_users, np.int64)
    users, idx = np.unique(u_imp, return_index=True)
    first_t[users] = ts[idx]
    users, idx = np.unique(u_imp[::-1], return_index=True)
    last_t[users] = ts[::-1][idx]

    n_imp = len(ts)
    first = np.zeros(n_imp, np.int64)
    end = np.zeros(n_imp, np.int64)
    parts_BC = []
    parts_BB = []
    parts_BT = []
    offset = 0  # how many actions are already in the store
    if len(cate_vocab) < 32000:
        cate_dtype = np.int16
    else:
        cate_dtype = np.int32
    imp_bucket = u_imp % n_buckets

    for k in range(n_buckets):
        table = pq.read_table(os.path.join(folder, f"b{k}.parquet"))
        u = table["u"].to_numpy().astype(np.int64)
        t = table["t"].to_numpy()

        # Sort by (user, time)
        order = np.lexsort((t, u))
        u = u[order]
        t = t[order]
        key = (u << 32) + t  # one sortable number per (user, time)

        users = np.unique(u)
        start_u = np.searchsorted(u, users)  # first action of each user
        end_first = np.searchsorted(key, (users << 32) + first_t[users])  # actions before the user's first impression
        end_last = np.searchsorted(key, (users << 32) + last_t[users])  # actions before the user's last impression
        lo = np.maximum(start_u, end_first - seq_len)  # nothing older than this can be in any window
        n = end_last - lo  # how many actions we keep per user
        new_start = offset + np.concatenate([[0], np.cumsum(n)[:-1]])  # where each user starts in the store
        keep = np.repeat(lo - (new_start - offset), n) + np.arange(n.sum())  # row numbers to keep, in order

        parts_BC.append(encode(table["c"].to_numpy()[order][keep], cate_vocab).astype(cate_dtype))
        parts_BB.append(encode(table["r"].to_numpy()[order][keep], brand_vocab).astype(np.int32))
        parts_BT.append(table["b"].to_numpy()[order][keep].astype(np.int8))

        # Pointers for the impressions of this bucket's users
        rows = np.where(imp_bucket == k)[0]
        iu = u_imp[rows]
        j = np.clip(np.searchsorted(users, iu), 0, len(users) - 1)
        has_history = users[j] == iu
        e = np.searchsorted(key, (iu << 32) + ts[rows])  # only actions BEFORE the impression
        end[rows] = np.where(has_history, e - lo[j] + new_start[j], 0)
        first[rows] = np.where(has_history, new_start[j], 0)

        offset += int(n.sum())
        del table, u, t, order, key, keep
        gc.collect()

    BC = join_chunks(parts_BC)
    BB = join_chunks(parts_BB)
    BT = join_chunks(parts_BT)
    size_gb = (BC.nbytes + BB.nbytes + BT.nbytes) / 1e9
    print(f"history store: {len(BT):,} actions ({size_gb:.1f} GB)")
    return first, end, BC, BB, BT


def get_history(dataset, rows):
    """The last seq_len actions before each impression in rows. Empty slots are 0.

    Returns three arrays of shape (len(rows), seq_len): categories, brands, action types.
    """
    seq_len = dataset.seq_len
    # For each row: the seq_len store positions just before its end pointer
    positions = dataset.end[rows, None] - seq_len + np.arange(seq_len)[None, :]
    # Positions before the user's first kept action are not valid (padding)
    valid = positions >= dataset.first[rows, None]
    positions = np.where(valid, positions, 0)
    hist_cate = np.where(valid, dataset.BC[positions], 0)
    hist_brand = np.where(valid, dataset.BB[positions], 0)
    hist_btag = np.where(valid, dataset.BT[positions], 0)
    return hist_cate, hist_brand, hist_btag


def save_checkpoint(path, first, end, BC, BB, BT):
    np.savez(path, first=first, end=end, BC=BC, BB=BB, BT=BT)


def load_checkpoint(path):
    h = np.load(path)
    return h["first"], h["end"], h["BC"], h["BB"], h["BT"]


def save_vocab(path, cate_vocab, brand_vocab):
    np.savez(path, cate=cate_vocab, brand=brand_vocab)


def load_vocab(path):
    v = np.load(path)
    return v["cate"], v["brand"]
