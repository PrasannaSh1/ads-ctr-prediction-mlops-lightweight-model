"""TwoStageScorer: the online two-stage cascade from notebook section 12, applied to serving.

How Pinterest serves the 3-tower (see evaluation/combine.py's docstring): the user tower runs
once per request; the cheap co-trained two-tower score (user_vec . ad_vec + dot_bias) ranks
every candidate from the ad cache; the cross tower + inter module (the full 3-tower) re-scores
only the top `cascade_top_frac`, the rest keep their two-tower score.

Two simplifications vs. true online serving, both deliberate and documented here rather than
silently wrong:
1. A user's "current" profile/history/click-rate are read from their most recent row in the
   offline dataset (`UserFeatureCache`) -- a real feature store would serve continuously
   updated values instead of a training-time snapshot.
2. The user x category click-rate (TE_CROSS) needs a (user, ad-category) join that this
   serving path does not precompute; it always uses the standardized prior (0.0) -- the same
   fallback features/click_rates.py uses for a key it has never seen.
"""
from __future__ import annotations

import numpy as np
import torch

from taobao_ctr.data.dataset import Dataset
from taobao_ctr.data.history import history
from taobao_ctr.models.layers import USER_FEATS
from taobao_ctr.models.towers import ThreeTower


class UserFeatureCache:
    """Per-user profile/history/click-rate model inputs, from that user's most recent row."""

    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        order = np.argsort(dataset.ts, kind="stable")
        last_row: dict[int, int] = {}
        for idx in order:  # later rows overwrite earlier ones -> ends with the most recent per user
            last_row[int(dataset.u[idx])] = int(idx)
        self._last_row_by_u_index = last_row
        self._u_index_by_raw_id = {int(raw): i + 1 for i, raw in enumerate(dataset.user_ids)}

    def row_for(self, raw_user_id: int) -> int:
        u_index = self._u_index_by_raw_id.get(int(raw_user_id))
        if u_index is None or u_index not in self._last_row_by_u_index:
            raise KeyError(f"user {raw_user_id} has no known impression history")
        return self._last_row_by_u_index[u_index]

    def batch_for(self, raw_user_id: int) -> dict:
        rows = np.array([self.row_for(raw_user_id)])
        d = self.dataset
        b = {key: torch.from_numpy(d.X[key][rows].astype(np.int64)) for key, _ in USER_FEATS}
        hc, hb, ht = history(rows, first=d.first, end=d.end, BC=d.BC, BB=d.BB, BT=d.BT, seq_len=d.seq_len)
        b["hist_cate"] = torch.from_numpy(hc.astype(np.int64))
        b["hist_brand"] = torch.from_numpy(hb.astype(np.int64))
        b["hist_btag"] = torch.from_numpy(ht.astype(np.int64))
        b["te"] = torch.from_numpy(d.TE[rows])
        return b


class TwoStageScorer:
    def __init__(
        self, model: ThreeTower, ad_cache: dict, user_cache: UserFeatureCache,
        device: torch.device, cascade_top_frac: float, stage2_chunk_size: int = 4096,
    ):
        self.model = model.to(device).eval()
        self.ad_cache, self.user_cache, self.device = ad_cache, user_cache, device
        self.cascade_top_frac = cascade_top_frac
        # The cross tower's attention allocates O(chunk x seq_len x tower_dim) per call; an
        # unbounded batch (e.g. 200K candidates) can exhaust RAM in one shot, so stage 2 always
        # runs in chunks -- the same thing a real server would have to do.
        self.stage2_chunk_size = stage2_chunk_size

    @torch.no_grad()
    def score(self, raw_user_id: int, candidate_ad_ids: np.ndarray) -> np.ndarray:
        """pCTR per candidate ad id, in the same order as `candidate_ad_ids`."""
        candidate_ad_ids = np.asarray(candidate_ad_ids)
        ad_ids = self.ad_cache["ad_ids"]
        idx = np.clip(np.searchsorted(ad_ids, candidate_ad_ids), 0, len(ad_ids) - 1)
        found = ad_ids[idx] == candidate_ad_ids
        if not found.all():
            raise KeyError(f"{int((~found).sum())} candidate ad id(s) not in the ad cache")

        ad_vecs = torch.from_numpy(self.ad_cache["ad_vecs"][idx]).to(self.device)
        cross_vecs = torch.from_numpy(self.ad_cache["cross_vecs"][idx]).to(self.device)

        user_batch = {k: v.to(self.device) for k, v in self.user_cache.batch_for(raw_user_id).items()}
        u_vec = self.model.user(user_batch)  # (1, tower_dim)
        stage1 = torch.sigmoid((u_vec * ad_vecs).sum(-1) + self.model.dot_bias).cpu().numpy()

        n = len(candidate_ad_ids)
        k = max(1, int(round(self.cascade_top_frac * n)))
        top = np.argsort(-stage1)[:k]

        hist_keys = ("hist_cate", "hist_brand", "hist_btag")
        out = stage1.copy()
        for s in range(0, k, self.stage2_chunk_size):
            chunk = top[s:s + self.stage2_chunk_size]
            m = len(chunk)
            b_repeated = {key: user_batch[key].expand(m, *user_batch[key].shape[1:]) for key in hist_keys}
            c = self.model.cross(cross_vecs[chunk], b_repeated)
            u_rep, a_rep = u_vec.expand(m, -1), ad_vecs[chunk]
            te_cross = torch.zeros(m, 1, device=self.device)  # standardized prior; see module docstring (2)
            rich = self.model.inter(torch.cat([u_rep, a_rep, c, te_cross], -1)) + self.model.task_bias(torch.cat([u_rep, a_rep], -1))
            out[chunk] = torch.sigmoid(rich.squeeze(-1)).cpu().numpy()
        return out
