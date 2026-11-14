"""The two-stage scorer used for serving.

For one request (one user, many candidate ads):
  1. run the user tower once,
  2. stage 1: score every candidate with the cheap dot product (user vector . cached ad vector),
  3. stage 2: re-score only the top cascade_top_frac with the full 3-tower (cross tower + inter module),
  4. the rest keep their stage-1 score.

Two simplifications compared to real online serving:
  1. A user's profile, history and click rate come from their most recent row in the dataset
     (UserFeatureCache). A real system would read live values from a feature store.
  2. The user x category click rate is not available at serving time, so it is set to 0.0,
     which is the standardised prior (the same fallback used for an unseen key in training).
"""
import numpy as np
import torch

from taobao_ctr.data.history import get_history
from taobao_ctr.models.layers import USER_FEATS


class UserFeatureCache:
    """Model inputs for a user, taken from that user's most recent impression."""

    def __init__(self, dataset):
        self.dataset = dataset
        # user number -> row of their most recent impression
        order = np.argsort(dataset.ts, kind="stable")
        last_row = {}
        for idx in order:  # later rows overwrite earlier ones
            last_row[int(dataset.u[idx])] = int(idx)
        self.last_row_by_user_number = last_row
        # raw user id -> user number (1..U)
        self.user_number_by_raw_id = {}
        for i, raw_id in enumerate(dataset.user_ids):
            self.user_number_by_raw_id[int(raw_id)] = i + 1

    def row_for(self, raw_user_id):
        user_number = self.user_number_by_raw_id.get(int(raw_user_id))
        if user_number is None or user_number not in self.last_row_by_user_number:
            raise KeyError(f"user {raw_user_id} has no known impression history")
        return self.last_row_by_user_number[user_number]

    def batch_for(self, raw_user_id):
        """A batch of size 1 with this user's inputs."""
        rows = np.array([self.row_for(raw_user_id)])
        d = self.dataset
        batch = {}
        for batch_key, _table_name in USER_FEATS:
            batch[batch_key] = torch.from_numpy(d.X[batch_key][rows].astype(np.int64))
        hist_cate, hist_brand, hist_btag = get_history(d, rows)
        batch["hist_cate"] = torch.from_numpy(hist_cate.astype(np.int64))
        batch["hist_brand"] = torch.from_numpy(hist_brand.astype(np.int64))
        batch["hist_btag"] = torch.from_numpy(hist_btag.astype(np.int64))
        batch["te"] = torch.from_numpy(d.TE[rows])
        return batch


class TwoStageScorer:
    def __init__(self, model, ad_cache, user_cache, device, cascade_top_frac, stage2_chunk_size=4096):
        self.model = model.to(device).eval()
        self.ad_cache = ad_cache
        self.user_cache = user_cache
        self.device = device
        self.cascade_top_frac = cascade_top_frac
        # Stage 2 runs in chunks: the cross tower's attention needs a lot of memory per candidate,
        # so scoring 200K candidates in one go could run out of RAM.
        self.stage2_chunk_size = stage2_chunk_size

    def score(self, raw_user_id, candidate_ad_ids):
        """Predicted CTR for every candidate ad, in the same order as candidate_ad_ids."""
        with torch.no_grad():
            candidate_ad_ids = np.asarray(candidate_ad_ids)

            # Find every candidate in the ad cache
            ad_ids = self.ad_cache["ad_ids"]
            idx = np.clip(np.searchsorted(ad_ids, candidate_ad_ids), 0, len(ad_ids) - 1)
            found = ad_ids[idx] == candidate_ad_ids
            if not found.all():
                n_missing = int((~found).sum())
                raise KeyError(f"{n_missing} candidate ad id(s) not in the ad cache")
            ad_vecs = torch.from_numpy(self.ad_cache["ad_vecs"][idx]).to(self.device)
            cross_vecs = torch.from_numpy(self.ad_cache["cross_vecs"][idx]).to(self.device)

            # Run the user tower once
            user_batch = {}
            for key, value in self.user_cache.batch_for(raw_user_id).items():
                user_batch[key] = value.to(self.device)
            u_vec = self.model.user(user_batch)  # shape (1, tower_dim)

            # Stage 1: the cheap score for every candidate
            stage1 = torch.sigmoid((u_vec * ad_vecs).sum(-1) + self.model.dot_bias).cpu().numpy()

            # Keep the top fraction
            n = len(candidate_ad_ids)
            k = max(1, int(round(self.cascade_top_frac * n)))
            top = np.argsort(-stage1)[:k]

            # Stage 2: the rich score for the top candidates, in chunks
            out = stage1.copy()
            for start in range(0, k, self.stage2_chunk_size):
                chunk = top[start:start + self.stage2_chunk_size]
                m = len(chunk)
                # The same user history, repeated once per candidate in the chunk
                history_batch = {
                    "hist_cate": user_batch["hist_cate"].expand(m, -1),
                    "hist_brand": user_batch["hist_brand"].expand(m, -1),
                    "hist_btag": user_batch["hist_btag"].expand(m, -1),
                }
                c = self.model.cross(cross_vecs[chunk], history_batch)
                u_rep = u_vec.expand(m, -1)
                a_rep = ad_vecs[chunk]
                te_cross = torch.zeros(m, 1, device=self.device)  # the standardised prior, see the note at the top
                rich = self.model.inter(torch.cat([u_rep, a_rep, c, te_cross], -1))
                rich = rich + self.model.task_bias(torch.cat([u_rep, a_rep], -1))
                out[chunk] = torch.sigmoid(rich.squeeze(-1)).cpu().numpy()
            return out
