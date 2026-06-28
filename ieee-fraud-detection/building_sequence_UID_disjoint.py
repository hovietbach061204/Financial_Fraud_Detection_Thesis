import numpy as np

def split_uids_by_row_count(uid_series, frac_train=0.8, seed=42):
    rng = np.random.default_rng(seed)

    # dropna=False to match your XGBoost split logic
    uid_counts = uid_series.value_counts(dropna=False)
    uids_arr = uid_counts.index.to_numpy()
    counts_arr = uid_counts.values.astype(np.int64)

    perm = rng.permutation(len(uids_arr))
    uids_shuf = uids_arr[perm]
    counts_shuf = counts_arr[perm]

    total = int(counts_shuf.sum())
    target = int(frac_train * total)
    cumsum = np.cumsum(counts_shuf)
    cut = int(np.searchsorted(cumsum, target)) + 1

    return (
        set(uids_shuf[:cut].tolist()),
        set(uids_shuf[cut:].tolist()),
        total,
        int(counts_shuf[:cut].sum()),
        int(counts_shuf[cut:].sum()),
    )

def build_per_uid_step_arrays(df, feature_cols, max_len,
                              uid_col="uid", time_col="DT", y_col="_y"):
    """
    v4 style:
      one sample = one UID
      X shape = (n_uids, MAX_LEN, n_features)
      Y shape = (n_uids, MAX_LEN), one label per real timestep
      M shape = (n_uids, MAX_LEN), True only for real timesteps
      L shape = (n_uids,), number of real timesteps kept
    """
    df = df.sort_values([uid_col, time_col], kind="mergesort")

    groups = list(df.groupby(uid_col, sort=False, dropna=False))
    n_uids = len(groups)
    n_features = len(feature_cols)

    X = np.zeros((n_uids, max_len, n_features), dtype=np.float32)
    L = np.zeros(n_uids, dtype=np.int64)
    Y = np.zeros((n_uids, max_len), dtype=np.float32)
    M = np.zeros((n_uids, max_len), dtype=bool)

    uids_out = []
    orig_out = []

    for i, (uid, g) in enumerate(groups):
        # Keep most recent MAX_LEN transactions for long UIDs.
        g_take = g.tail(max_len)
        take = len(g_take)

        X[i, :take, :] = g_take[feature_cols].to_numpy(dtype=np.float32)
        Y[i, :take] = g_take[y_col].to_numpy(dtype=np.float32)
        M[i, :take] = True
        L[i] = take

        uids_out.append(uid)
        orig_out.append(g_take.index.to_numpy())

    return X, L, Y, M, np.array(uids_out, dtype=object), orig_out


def build_current_txn_windows(df, feature_cols, max_len,
                              uid_col='uid', time_col='DT', y_col='_y'):
    # v6 version
    """One sample per transaction: right-padded UID history ending at that row."""
    df_sorted = df.sort_values([uid_col, time_col], kind='mergesort')
    feat = df_sorted[feature_cols].to_numpy(dtype=np.float32)
    y = df_sorted[y_col].to_numpy(dtype=np.float32)
    uids = df_sorted[uid_col].to_numpy()
    orig = df_sorted.index.to_numpy()

    n = len(df_sorted)
    F = len(feature_cols)
    X = np.zeros((n, max_len, F), dtype=np.float32)
    L = np.zeros(n, dtype=np.int64)
    y_out = y.copy()
    orig_out = orig.copy()
    uid_out = uids.copy()

    boundaries = np.flatnonzero(np.concatenate([[True], uids[1:] != uids[:-1]]))
    boundaries = np.append(boundaries, n)

    for b_start, b_end in zip(boundaries[:-1], boundaries[1:]):
        block = feat[b_start:b_end]
        for t in range(b_end - b_start):
            row = b_start + t
            start = max(0, t - max_len + 1)
            seq = block[start:t + 1]
            X[row, :seq.shape[0], :] = seq
            L[row] = seq.shape[0]

    return X, L, y_out, orig_out, uid_out
