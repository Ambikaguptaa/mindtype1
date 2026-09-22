"""
infer_live.py
---------------
Scores typing sessions in real-time or from simulation:
  1. Receives captured or simulated per-keystroke feature frame.
  2. Runs feature engineering, personal baseline deviation, and robust statistics.
  3. Evaluates sequence with the trained LSTM model.
  4. Conducts nonparametric statistical checks (z-score hesitation sweep, Mann-Whitney U test).
  5. Computes behavioral feature contributions and logs results.

Run directly for quick developer testing: `python src/infer_live.py`
"""
import sys, os, json, datetime
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
from scipy import stats
from tensorflow import keras

from features import compute_user_baselines, extract_comprehensive_features, FEATURES, SEQ_LEN
from generate_data import simulate_session
from models_engine import compute_feature_contributions
import dsp

LOG_PATH = "data/session_log.csv"
HESITATION_Z_THRESHOLD = 2.0


def make_live_session(user_id: str, label_hint: int, rng: np.random.Generator):
    """Generates one simulated session for testing the offline/online scoring pipeline."""
    baselines_path = "models/baselines_plain.csv"
    if os.path.exists(baselines_path):
        baselines = pd.read_csv(baselines_path)
        row = baselines[baselines.user_id == user_id]
        if row.empty:
            user_speed, user_var = 180.0, 45.0
        else:
            user_speed, user_var = row.iloc[0]["baseline_mean_ms"], row.iloc[0]["baseline_std_ms"]
    else:
        user_speed, user_var = 180.0, 45.0

    ikt, dwell, bksp, err = simulate_session(user_speed, user_var, label_hint)
    return pd.DataFrame({
        "user_id": user_id,
        "session_id": f"{user_id}_live_{rng.integers(1_000_000)}",
        "label": label_hint,
        "key_index": range(len(ikt)),
        "inter_key_ms": ikt,
        "dwell_ms": dwell,
        "is_backspace": bksp,
        "is_error": err,
    })


def detect_hesitation_spikes(feat_df: pd.DataFrame, z_thresh: float = HESITATION_Z_THRESHOLD) -> dict:
    """Detects statistical hesitation anomalies on this passage's own keystroke
    rhythm using scipy.stats.zscore. Identifies pauses clustering adjacent to
    typing errors.
    """
    ikt = feat_df["inter_key_ms"].to_numpy(dtype=float)
    is_error = feat_df["is_error"].to_numpy(dtype=int) if "is_error" in feat_df else np.zeros(len(ikt), dtype=int)

    if len(ikt) < 5 or np.std(ikt) < 1e-6:
        return {"hesitation_spikes": 0, "hesitation_near_error": 0, "rhythm_cv": 0.0}

    z = stats.zscore(ikt)
    z = np.nan_to_num(z, nan=0.0)
    spike_positions = np.where(z >= z_thresh)[0]

    near_error = 0
    for pos in spike_positions:
        lo, hi = max(0, pos - 1), min(len(is_error) - 1, pos + 1)
        if is_error[lo:hi + 1].sum() > 0:
            near_error += 1

    rhythm_cv = float(stats.variation(ikt, nan_policy="omit"))
    rhythm_cv = 0.0 if np.isnan(rhythm_cv) else round(rhythm_cv, 3)

    return {
        "hesitation_spikes": int(len(spike_positions)),
        "hesitation_near_error": int(near_error),
        "rhythm_cv": rhythm_cv,
    }


def compare_rhythm_to_baseline(passage_ikt: np.ndarray, baseline_ikt: np.ndarray) -> dict:
    """Nonparametric Mann-Whitney U test comparing this passage's inter-key timing
    distribution against the personal calibration baseline distribution.
    """
    baseline_ikt = np.asarray(baseline_ikt, dtype=float)
    passage_ikt = np.asarray(passage_ikt, dtype=float)
    if len(baseline_ikt) < 5 or len(passage_ikt) < 5:
        return {"rhythm_shift_p": None, "rhythm_shift_significant": None}
    try:
        u_stat, p_value = stats.mannwhitneyu(passage_ikt, baseline_ikt, alternative="two-sided")
        return {
            "rhythm_shift_u": round(float(u_stat), 1),
            "rhythm_shift_p": round(float(p_value), 4),
            "rhythm_shift_significant": bool(p_value < 0.05),
        }
    except Exception:
        return {"rhythm_shift_p": None, "rhythm_shift_significant": None}


_model_cache = {}


def _get_model():
    if "model" not in _model_cache:
        model_path = "models/lstm_model.keras"
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Trained model not found at '{model_path}'. Run "
                "`python src/generate_data.py && python src/train.py` first."
            )
        _model_cache["model"] = keras.models.load_model(model_path)
    return _model_cache["model"]


def score_feature_frame(
    feat_df: pd.DataFrame,
    b_mean: float,
    b_std: float,
    baseline_ikt: np.ndarray = None,
    baseline_stats: dict = None
) -> dict:
    """Core scorer evaluating a passage's keystroke feature frame.
    Returns model probability, Behavioral Strain score, robust summary metrics,
    statistical tests, and feature contributions.
    """
    if feat_df is None or len(feat_df) == 0:
        raise ValueError("score_feature_frame received an empty feature frame.")

    feat_df = feat_df.copy().sort_values("key_index")
    b_std = b_std if b_std and b_std > 1e-6 else 25.0
    feat_df["baseline_dev"] = (feat_df["inter_key_ms"] - b_mean) / b_std

    # Prepare sequence for LSTM
    arr = feat_df[FEATURES].values.astype(np.float32)
    n = len(arr)
    if n >= SEQ_LEN:
        arr = arr[:SEQ_LEN]
        n = SEQ_LEN
    else:
        pad = np.zeros((SEQ_LEN - n, len(FEATURES)), dtype=np.float32)
        arr = np.vstack([arr, pad])

    scaler_path = "models/scaler.npz"
    if os.path.exists(scaler_path):
        scaler = np.load(scaler_path)
        s_mean, s_std = scaler["mean"], scaler["std"]
    else:
        s_mean = np.array([200.0, 80.0, 0.05, 0.04, 0.0], dtype=np.float32)
        s_std = np.array([120.0, 40.0, 0.20, 0.20, 1.0], dtype=np.float32)

    X = ((arr - s_mean) / s_std)[None, ...].astype(np.float32)
    X[0, n:] = 0.0

    model = _get_model()
    prob = float(model.predict(X, verbose=0).ravel()[0])

    # Robust summary features
    comp_feats = extract_comprehensive_features(feat_df, baseline_stats)

    # Statistical checks
    stats_block = detect_hesitation_spikes(feat_df)
    if baseline_ikt is not None:
        stats_block.update(compare_rhythm_to_baseline(feat_df["inter_key_ms"].to_numpy(), baseline_ikt))
    else:
        stats_block.setdefault("rhythm_shift_p", None)
        stats_block.setdefault("rhythm_shift_significant", None)

    strain_score = round(prob * 100, 1)
    if strain_score < 35.0:
        pred_state = "Baseline / Relaxed"
    elif strain_score < 65.0:
        pred_state = "Mild Behavioral Shift"
    else:
        pred_state = "Elevated Strain Pattern"

    result = {
        "strain_probability": round(prob, 4),
        "strain_score": strain_score,
        "predicted_state": pred_state,
        "avg_typing_speed_ms": round(float(feat_df["inter_key_ms"].mean()), 1),
        "backspace_rate": round(float(feat_df["is_backspace"].mean()), 3),
        "error_rate": round(float(feat_df["is_error"].mean()), 3),
        "baseline_deviation": round(float(feat_df["baseline_dev"].mean()), 2),
        "n_keystrokes": int(len(feat_df)),
        **comp_feats,
        **stats_block,
    }

    # Behavioral explainability contributions
    result["feature_contributions"] = compute_feature_contributions(result, baseline_stats)
    return result


def score_session(session_df: pd.DataFrame) -> dict:
    if session_df is None or len(session_df) == 0:
        raise ValueError("score_session received an empty session.")

    baselines_path = "models/baselines_plain.csv"
    if os.path.exists(baselines_path):
        baselines = pd.read_csv(baselines_path)
        b_row = baselines[baselines.user_id == session_df.user_id.iloc[0]]
        if b_row.empty:
            b_mean, b_std = float(baselines.baseline_mean_ms.mean()), float(baselines.baseline_std_ms.mean())
        else:
            b_mean, b_std = float(b_row.iloc[0]["baseline_mean_ms"]), float(b_row.iloc[0]["baseline_std_ms"])
    else:
        b_mean, b_std = 180.0, 45.0

    core = score_feature_frame(session_df, b_mean, b_std)
    return {
        "user_id": session_df.user_id.iloc[0],
        "user_id_pseudonymized": dsp.pseudonymize_id(session_df.user_id.iloc[0]),
        "session_id": session_df.session_id.iloc[0],
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        **core,
    }


def log_result(result: dict):
    """Appends scored session results to CSV log."""
    os.makedirs("data", exist_ok=True)
    clean_dict = {k: v for k, v in result.items() if not isinstance(v, (list, dict, np.ndarray))}
    df_new = pd.DataFrame([clean_dict])
    try:
        if os.path.exists(LOG_PATH):
            existing_cols = pd.read_csv(LOG_PATH, nrows=0).columns.tolist()
            all_cols = existing_cols + [c for c in df_new.columns if c not in existing_cols]
            df_new = df_new.reindex(columns=all_cols)
            df_new.to_csv(LOG_PATH, mode="a", header=False, index=False)
        else:
            df_new.to_csv(LOG_PATH, index=False)
    except Exception as e:
        print(f"[infer_live] Warning: could not write to session log: {e}")


def plain_language_report(result: dict) -> str:
    score = result.get("strain_score", 50.0)
    if score < 35:
        body = "Typing rhythm corresponds closely to the calibrated baseline. No significant indicators of strain."
    elif score < 65:
        body = "Moderate behavioral variation observed — slight speed drift and occasional hesitation pauses under challenging conditions."
    else:
        body = "Measurable changes observed: widened inter-key gaps, higher pause frequency, and increased correction activity."
    return (
        f"MindType Analysis: Behavioral Strain Index {score}/100 ({result.get('predicted_state', 'Standard')}).\n"
        f"{body}\n"
        "This is an experimental behavioral estimate of typing dynamics — not a medical diagnosis."
    )


if __name__ == "__main__":
    rng = np.random.default_rng()
    uid = sys.argv[1] if len(sys.argv) > 1 else "user_003"
    lbl = int(sys.argv[2]) if len(sys.argv) > 2 else int(rng.integers(0, 2))

    session = make_live_session(uid, lbl, rng)
    res = score_session(session)
    log_result(res)

    print(json.dumps({k: v for k, v in res.items() if not isinstance(v, (list, dict))}, indent=2))
    print("\n" + plain_language_report(res))
