"""
features.py  (DE — Data Engineering)
-------------------------------------
Turns raw per-keystroke rows into:
  1. Sequential per-keystroke feature vectors for the LSTM (FEATURES),
     including deviation from personal baseline.
  2. Comprehensive passage-level summary features incorporating robust
     statistics (median, MAD, CV, percentiles, pause frequencies, burst patterns,
     temporal trends, and dwell dynamics).
  3. Personal baseline calibration metrics from resting/neutral sessions.
"""
import numpy as np
import pandas as pd
from scipy import stats

SEQ_LEN = 220  # max keystrokes per sequence for LSTM masking
FEATURES = ["inter_key_ms", "dwell_ms", "is_backspace", "is_error", "baseline_dev"]


def compute_user_baselines(df: pd.DataFrame) -> pd.DataFrame:
    """Baseline = each user's typical rhythm, estimated from their calm
    (label == 0) sessions only."""
    calm = df[df["label"] == 0]
    baseline = calm.groupby("user_id")["inter_key_ms"].agg(["mean", "std"]).reset_index()
    baseline.columns = ["user_id", "baseline_mean_ms", "baseline_std_ms"]
    baseline["baseline_std_ms"] = baseline["baseline_std_ms"].clip(lower=5.0)
    return baseline


def build_sequences(df: pd.DataFrame, baselines: pd.DataFrame):
    """Returns X (n_sessions, SEQ_LEN, n_features), y (n_sessions,), and
    session_ids in matching order for sequential neural network modeling."""
    merged = df.merge(baselines, on="user_id", how="left")
    pop_mean = baselines["baseline_mean_ms"].mean() if not baselines.empty else 180.0
    pop_std = baselines["baseline_std_ms"].mean() if not baselines.empty else 45.0
    merged["baseline_mean_ms"] = merged["baseline_mean_ms"].fillna(pop_mean)
    merged["baseline_std_ms"] = merged["baseline_std_ms"].fillna(pop_std)

    merged["baseline_dev"] = (
        (merged["inter_key_ms"] - merged["baseline_mean_ms"]) / merged["baseline_std_ms"]
    )

    X, y, sids, lengths = [], [], [], []
    for sid, g in merged.sort_values("key_index").groupby("session_id"):
        g = g.sort_values("key_index")
        arr = g[FEATURES].values.astype(np.float32)
        n = len(arr)
        if n >= SEQ_LEN:
            arr = arr[:SEQ_LEN]
            n = SEQ_LEN
        else:
            pad = np.zeros((SEQ_LEN - n, len(FEATURES)), dtype=np.float32)
            arr = np.vstack([arr, pad])
        X.append(arr)
        y.append(g["label"].iloc[0])
        sids.append(sid)
        lengths.append(n)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), sids, np.array(lengths)


def fit_scaler_masked(X: np.ndarray, lengths: np.ndarray):
    """Standardization ignoring zero-padded timesteps."""
    rows = []
    for i, n in enumerate(lengths):
        rows.append(X[i, :n])
    flat = np.vstack(rows)
    mean = flat.mean(axis=0)
    std = flat.std(axis=0).clip(min=1e-6)
    return mean, std


def apply_scaler(X: np.ndarray, mean: np.ndarray, std: np.ndarray, lengths: np.ndarray = None):
    """Scales real timesteps while preserving exact zero-padding for masking."""
    scaled = (X - mean) / std
    if lengths is not None:
        for i, n in enumerate(lengths):
            if n < X.shape[1]:
                scaled[i, n:] = 0.0
    return scaled


# =====================================================================
# COMPREHENSIVE BEHAVIORAL FEATURE EXTRACTION (ROBUST STATISTICS)
# =====================================================================

def extract_comprehensive_features(feat_df: pd.DataFrame, baseline_stats: dict = None) -> dict:
    """Computes a rich suite of robust summary statistics, timing dynamics,
    pause structures, and correction patterns from a passage's feature frame.
    
    Using robust statistics (median, MAD, trimmed ranges) prevents a single
    isolated long pause from distorting the entire behavioral profile.
    """
    if feat_df is None or len(feat_df) == 0:
        return {}

    ikt = feat_df["inter_key_ms"].to_numpy(dtype=float)
    dwell = feat_df["dwell_ms"].to_numpy(dtype=float) if "dwell_ms" in feat_df else np.zeros(len(ikt))
    bksp = feat_df["is_backspace"].to_numpy(dtype=int) if "is_backspace" in feat_df else np.zeros(len(ikt), dtype=int)
    err = feat_df["is_error"].to_numpy(dtype=int) if "is_error" in feat_df else np.zeros(len(ikt), dtype=int)
    n = len(ikt)

    # 1. Central Tendency & Dispersion (Inter-Key Timing)
    ikt_mean = float(np.mean(ikt))
    ikt_median = float(np.median(ikt))
    ikt_std = float(np.std(ikt))
    ikt_cv = float(ikt_std / (ikt_mean + 1e-6))
    
    # Median Absolute Deviation (MAD)
    ikt_mad = float(stats.median_abs_deviation(ikt, scale="normal")) if n > 2 else 0.0

    # Percentiles
    p10 = float(np.percentile(ikt, 10))
    p25 = float(np.percentile(ikt, 25))
    p75 = float(np.percentile(ikt, 75))
    p90 = float(np.percentile(ikt, 90))
    iqr = p75 - p25

    # 2. Pause Dynamics
    pauses_500ms = int(np.sum(ikt >= 500))
    pauses_1000ms = int(np.sum(ikt >= 1000))
    pauses_2000ms = int(np.sum(ikt >= 2000))
    pauses_4000ms = int(np.sum(ikt >= 4000))
    longest_pause_ms = float(np.max(ikt)) if n > 0 else 0.0
    pause_frequency_pct = float((pauses_500ms / max(n, 1)) * 100)

    # 3. Dwell Dynamics
    dwell_mean = float(np.mean(dwell))
    dwell_median = float(np.median(dwell))
    dwell_std = float(np.std(dwell))
    dwell_var = float(np.var(dwell))
    dwell_to_ikt_ratio = float(dwell_mean / (ikt_mean + 1e-6))

    # 4. Correction & Error Dynamics
    backspace_rate = float(np.mean(bksp))
    error_rate = float(np.mean(err))

    # Correction Bursts (consecutive backspaces)
    burst_count = 0
    max_consecutive_bksp = 0
    current_burst = 0
    for b in bksp:
        if b == 1:
            current_burst += 1
            if current_burst == 2:
                burst_count += 1
            max_consecutive_bksp = max(max_consecutive_bksp, current_burst)
        else:
            current_burst = 0

    # Error Clusters (errors within 2 keys of each other)
    error_positions = np.where(err == 1)[0]
    error_cluster_count = 0
    if len(error_positions) > 1:
        diffs = np.diff(error_positions)
        error_cluster_count = int(np.sum(diffs <= 2))

    # 5. Temporal Trend (First-half vs Second-half pacing)
    half_idx = n // 2
    if half_idx >= 4:
        first_half_speed = float(np.mean(ikt[:half_idx]))
        second_half_speed = float(np.mean(ikt[half_idx:]))
        speed_drift_pct = float(((second_half_speed - first_half_speed) / (first_half_speed + 1e-6)) * 100)
    else:
        first_half_speed = ikt_mean
        second_half_speed = ikt_mean
        speed_drift_pct = 0.0

    # 6. Autocorrelation (Lag-1 rhythm predictability)
    if n > 6:
        try:
            autocorr_1 = float(pd.Series(ikt).autocorr(lag=1))
            if np.isnan(autocorr_1):
                autocorr_1 = 0.0
        except Exception:
            autocorr_1 = 0.0
    else:
        autocorr_1 = 0.0

    # 7. Speed Estimates
    # Words per minute: standard definition is (characters / 5) / minutes
    total_time_sec = float(np.sum(ikt)) / 1000.0
    cps = float(n / max(total_time_sec, 0.5))
    wpm = float(cps * 12.0)  # 60 / 5 = 12

    # Assemble summary record
    record = {
        "n_keystrokes": n,
        "total_time_sec": round(total_time_sec, 1),
        "wpm": round(wpm, 1),
        "chars_per_sec": round(cps, 2),
        "inter_key_mean_ms": round(ikt_mean, 1),
        "inter_key_median_ms": round(ikt_median, 1),
        "inter_key_std_ms": round(ikt_std, 1),
        "inter_key_cv": round(ikt_cv, 3),
        "inter_key_mad_ms": round(ikt_mad, 1),
        "p10_ms": round(p10, 1),
        "p25_ms": round(p25, 1),
        "p75_ms": round(p75, 1),
        "p90_ms": round(p90, 1),
        "iqr_ms": round(iqr, 1),
        "dwell_mean_ms": round(dwell_mean, 1),
        "dwell_median_ms": round(dwell_median, 1),
        "dwell_std_ms": round(dwell_std, 1),
        "dwell_variance": round(dwell_var, 1),
        "dwell_to_ikt_ratio": round(dwell_to_ikt_ratio, 3),
        "pauses_500ms": pauses_500ms,
        "pauses_1000ms": pauses_1000ms,
        "pauses_2000ms": pauses_2000ms,
        "pauses_4000ms": pauses_4000ms,
        "pause_frequency_pct": round(pause_frequency_pct, 1),
        "longest_pause_ms": round(longest_pause_ms, 1),
        "backspace_rate": round(backspace_rate, 4),
        "error_rate": round(error_rate, 4),
        "correction_bursts": burst_count,
        "max_consecutive_backspaces": max_consecutive_bksp,
        "error_clusters": error_cluster_count,
        "speed_drift_pct": round(speed_drift_pct, 1),
        "autocorr_lag1": round(autocorr_1, 3),
    }

    # 8. Relative Deltas vs Personal Baseline
    if baseline_stats:
        b_mean = baseline_stats.get("inter_key_mean_ms", ikt_mean)
        b_cv = baseline_stats.get("inter_key_cv", ikt_cv)
        b_pause_freq = baseline_stats.get("pause_frequency_pct", pause_frequency_pct)
        b_bksp = baseline_stats.get("backspace_rate", backspace_rate)

        speed_pct_diff = ((ikt_mean - b_mean) / (b_mean + 1e-6)) * 100.0
        cv_pct_diff = ((ikt_cv - b_cv) / (b_cv + 1e-6)) * 100.0
        pause_pct_diff = ((pause_frequency_pct - b_pause_freq) / (b_pause_freq + 1e-6)) * 100.0
        bksp_pct_diff = ((backspace_rate - b_bksp) / (b_bksp + 1e-6)) * 100.0

        record.update({
            "delta_speed_pct": round(speed_pct_diff, 1),
            "delta_cv_pct": round(cv_pct_diff, 1),
            "delta_pause_pct": round(pause_pct_diff, 1),
            "delta_backspace_pct": round(bksp_pct_diff, 1),
        })

    return record
