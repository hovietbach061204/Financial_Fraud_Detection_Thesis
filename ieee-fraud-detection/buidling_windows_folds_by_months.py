import numpy as np

def build_uid_windows(df, feature_cols, window, uid_col='uid', time_col='DT'):
    df_sorted = df.sort_values([uid_col, time_col], kind='mergesort')
    feat = df_sorted[feature_cols].to_numpy(dtype=np.float32)
    uids = df_sorted[uid_col].to_numpy()
    orig = df_sorted.index.to_numpy()

    n = len(df_sorted); f = len(feature_cols)
    X = np.zeros((n, window, f), dtype=np.float32)
    L = np.zeros(n, dtype=np.int64)        # NEW: real-step count per sample

    boundaries = np.flatnonzero(np.concatenate([[True], uids[1:] != uids[:-1]]))
    boundaries = np.append(boundaries, n)
    for b_start, b_end in zip(boundaries[:-1], boundaries[1:]):
        block = feat[b_start:b_end]
        for t in range(b_end - b_start):
            start = max(0, t - window + 1)
            seq = block[start:t + 1]
            X[b_start + t, -seq.shape[0]:] = seq
            L[b_start + t] = seq.shape[0]
    return X, orig, L