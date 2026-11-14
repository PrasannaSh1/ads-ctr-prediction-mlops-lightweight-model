"""Dataset: one object that holds everything the models train on.

It is built by pipeline.load_dataset() from the files in artifacts/cache.
"""


class Dataset:
    def __init__(self, X, y, ts, u, user_ids, TE, first, end, BC, BB, BT, sizes, seq_len):
        self.X = X                # dict: id column name -> numpy array of encoded ids (one per impression)
        self.y = y                # label per impression: 1 = clicked, 0 = not clicked
        self.ts = ts              # timestamp per impression
        self.u = u                # user number per impression (1..U)
        self.user_ids = user_ids  # raw user id for each user number: user_ids[u - 1]
        self.TE = TE              # the 8 click-rate features per impression
        self.first = first        # history store: where this impression's user starts
        self.end = end            # history store: just after the last action before this impression
        self.BC = BC              # history store: category of every kept action
        self.BB = BB              # history store: brand of every kept action
        self.BT = BT              # history store: action type of every kept action
        self.sizes = sizes        # dict: id column name -> how many distinct values it has
        self.seq_len = seq_len    # how many past actions we look at (50)
