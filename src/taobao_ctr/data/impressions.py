"""Loads the impressions (raw_sample.csv) with the ad and user tables joined on, turns every
id column into small numbers 1..K (0 = unknown), and splits the rows by time."""
import numpy as np
import pandas as pd

from taobao_ctr.data.download import kaggle_csv

# The 8 user profile columns
USER_COLS = [
    "cms_segid", "cms_group_id", "final_gender_code", "age_level",
    "pvalue_level", "shopping_level", "occupation", "new_user_class_level",
]
# Context columns: the ad slot on the page, and the hour of the day
CTX_COLS = ["pid", "hour"]
# Ad id columns
AD_ID_COLS = ["adgroup_id", "campaign_id", "customer"]
# Every id column the models get, in this order
ID_COLS = USER_COLS + CTX_COLS + AD_ID_COLS + ["ad_cate", "ad_brand", "price_bucket"]


def encode(raw_values, vocab):
    """Turn raw ids into numbers 1..K. Unknown or missing ids (-1) become 0.

    vocab must be a sorted numpy array of all the known ids.
    """
    raw_values = np.asarray(raw_values)
    # Where would each value sit in the sorted vocab?
    position = np.searchsorted(vocab, raw_values)
    position = np.clip(position, 0, len(vocab) - 1)
    # Is the value really in the vocab, and not a missing value?
    is_known = (raw_values >= 0) & (vocab[position] == raw_values)
    encoded = np.where(is_known, position + 1, 0)
    return encoded


def lookup_column(table, row_numbers, column):
    """One value per impression, read from a joined table. Rows with -1 (no match) become NaN."""
    values = table[column].values.astype("float64")
    return np.where(row_numbers >= 0, values[row_numbers], np.nan)


def load_impressions(data_dir, kaggle_dataset):
    """Read the 3 CSV files and return a dictionary with everything about the impressions."""
    # 1. The impressions: one row = one ad shown to one user
    raw_path = kaggle_csv("raw_sample.csv", data_dir, kaggle_dataset)
    raw = pd.read_csv(
        raw_path,
        usecols=["user", "time_stamp", "adgroup_id", "pid", "clk"],
        dtype={"user": "int32", "time_stamp": "int64", "adgroup_id": "int32", "pid": "category", "clk": "int8"},
    )
    # Sort by time, because the train / validation / test split is by time
    raw = raw.sort_values("time_stamp", kind="stable").reset_index(drop=True)

    # 2. The ad table: one row per ad
    ads_path = kaggle_csv("ad_feature.csv", data_dir, kaggle_dataset)
    ads = pd.read_csv(ads_path).set_index("adgroup_id")

    # 3. The user table: one row per user
    users_path = kaggle_csv("user_profile.csv", data_dir, kaggle_dataset)
    users = pd.read_csv(users_path)
    users.columns = users.columns.str.strip()  # one column name has a space at the end
    users = users.rename(columns={"userid": "user"})
    users = users.drop_duplicates("user").set_index("user")

    timestamps = raw["time_stamp"].values

    # Give every user a number from 1 to U
    user_ids, user_number = np.unique(raw["user"].values, return_inverse=True)
    user_number = user_number.ravel().astype(np.int64) + 1

    # For each impression: which row of the ad table and of the user table? (-1 = not found)
    ad_row = ads.index.get_indexer(raw["adgroup_id"].values)
    user_row = users.index.get_indexer(raw["user"].values)

    # Collect the raw values of every id column. Floats, so NaN can mean "missing".
    columns = []
    for col in USER_COLS:
        columns.append((col, lookup_column(users, user_row, col)))
    for col in ["campaign_id", "customer"]:
        columns.append((col, lookup_column(ads, ad_row, col)))
    columns.append(("pid", raw["pid"].cat.codes.values.astype("float64")))
    hour = ((timestamps + 8 * 3600) // 3600) % 24  # China time is UTC+8
    columns.append(("hour", hour.astype("float64")))
    columns.append(("adgroup_id", raw["adgroup_id"].values.astype("float64")))

    # Encode every column to 1..K and remember how many different values it has
    X = {}
    sizes = {}
    for col, values in columns:
        not_missing = ~np.isnan(values)
        vocab = np.unique(values[not_missing])
        X[col] = encode(values, vocab).astype(np.int32)
        sizes[col] = len(vocab)

    # Price: put it into 39 buckets on a log scale
    price = lookup_column(ads, ad_row, "price")
    price = np.nan_to_num(price)
    price_bucket = np.floor(np.log1p(price) * 2)
    price_bucket = np.clip(price_bucket, 0, 38) + 1
    X["price_bucket"] = price_bucket.astype(np.int32)

    click_rate = raw["clk"].mean()
    print(f"impressions {len(raw):,} | users {len(user_ids):,} | CTR {click_rate:.4f}")

    result = {
        "ts": timestamps,
        "y": raw["clk"].values.astype(np.float32),
        "u": user_number,
        "user_ids": user_ids,
        "X": X,
        "sizes": sizes,
        "price": price,
        "ad_cate": lookup_column(ads, ad_row, "cate_id"),
        "ad_brand": lookup_column(ads, ad_row, "brand"),
    }
    return result


def split(timestamps):
    """Split the rows by time: the last day is test, the last 10% of the rest is validation."""
    one_day = 86400
    test_start = timestamps.max() - one_day
    train_rows = np.where(timestamps < test_start)[0]
    test_rows = np.where(timestamps >= test_start)[0]
    n_val = int(0.1 * len(train_rows))
    fit_rows = train_rows[:-n_val]
    val_rows = train_rows[-n_val:]
    return train_rows, fit_rows, val_rows, test_rows
