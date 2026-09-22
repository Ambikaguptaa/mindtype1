"""
reliability.py
--------------
Scientific reliability and session quality evaluation engine for MindType.

Calculates:
  1. Session Quality Score (0-100): granular evaluation of browser focus,
     paste attempts, tab switches, idle interruptions, and keystroke volume.
  2. Measurement Reliability Score (0-100): comprehensive psychometric and
     statistical validity score assessing baseline stability, cross-passage
     consistency across task conditions (Neutral vs Load), and signal-to-noise ratio.
  3. Quality & Reliability Guardrails: suppresses strong behavioral conclusions
     if data quality does not meet research thresholds.
"""
from typing import List, Dict, Tuple
import numpy as np


def compute_passage_quality(quality_meta: dict, n_keystrokes: int, expected_chars: int) -> dict:
    """Evaluates the execution quality of a single passage.
    Returns a score (0-100), rating, and granular audit factors.
    """
    if not quality_meta:
        quality_meta = {}

    paste_attempts = quality_meta.get("paste_attempts", 0)
    blur_count = quality_meta.get("blur_count", 0)
    visibility_changes = quality_meta.get("visibility_changes", 0)
    total_unfocused_ms = quality_meta.get("total_unfocused_ms", 0.0)
    idle_interruptions = quality_meta.get("idle_interruptions", 0)

    score = 100.0
    deductions = []

    # 1. Paste attempts: severe penalty (pasting invalidates keystroke timing)
    if paste_attempts > 0:
        penalty = min(paste_attempts * 35.0, 70.0)
        score -= penalty
        deductions.append(f"{paste_attempts} paste attempt(s) detected (-{penalty:.0f} pts)")

    # 2. Window blur & tab switching
    focus_loss_count = blur_count + visibility_changes
    if focus_loss_count > 0:
        penalty = min(focus_loss_count * 8.0, 30.0)
        score -= penalty
        deductions.append(f"{focus_loss_count} focus disruption(s) / tab switch (-{penalty:.0f} pts)")

    # 3. Unfocused time
    if total_unfocused_ms > 3000:
        penalty = min((total_unfocused_ms / 1000.0) * 2.0, 20.0)
        score -= penalty
        deductions.append(f"Window was unfocused for {total_unfocused_ms/1000.0:.1f}s (-{penalty:.0f} pts)")

    # 4. Long idle interruptions (>4.5s)
    if idle_interruptions > 0:
        penalty = min(idle_interruptions * 6.0, 20.0)
        score -= penalty
        deductions.append(f"{idle_interruptions} prolonged mid-passage pause(s) (-{penalty:.0f} pts)")

    # 5. Keystroke volume completeness
    completeness_ratio = n_keystrokes / max(expected_chars * 0.8, 1)
    if completeness_ratio < 0.85:
        penalty = (1.0 - completeness_ratio) * 40.0
        score -= penalty
        deductions.append(f"Incomplete keystroke sample ({n_keystrokes} keys vs {expected_chars} expected)")

    score = max(0.0, min(100.0, score))

    if score >= 80.0:
        tier = "High"
    elif score >= 55.0:
        tier = "Moderate"
    else:
        tier = "Low"

    factors = [
        {"name": "Focus Maintained", "status": focus_loss_count == 0, "detail": f"{focus_loss_count} interruptions"},
        {"name": "No Paste Detected", "status": paste_attempts == 0, "detail": f"{paste_attempts} paste attempts"},
        {"name": "Typing Continuity", "status": idle_interruptions <= 1, "detail": f"{idle_interruptions} long pauses"},
        {"name": "Keystroke Sample Volume", "status": completeness_ratio >= 0.85, "detail": f"{n_keystrokes} keystrokes"},
    ]

    return {
        "quality_score": round(score, 1),
        "tier": tier,
        "factors": factors,
        "deductions": deductions,
        "paste_attempts": paste_attempts,
        "focus_loss_count": focus_loss_count,
        "total_unfocused_sec": round(total_unfocused_ms / 1000.0, 1),
    }


def compute_session_reliability(
    passage_results: List[Dict],
    passages_meta: List[Dict]
) -> Dict:
    """Computes overall session Measurement Reliability (0-100) and overall Session Quality.
    
    Evaluates:
      - Overall Session Quality across all passages (average quality & minimum quality)
      - Completeness: full 5-stage assessment execution
      - Baseline stability: Passage 1 calibration variance and keystroke count
      - Condition consistency:
          * Do Neutral tasks (P2 and P4) demonstrate comparable stability?
          * Do Load tasks (P3 and P5) exhibit consistent cognitive load dynamics?
      - Model and statistical agreement across diagnostic passages
    """
    total_expected = len(passages_meta)
    completed = len(passage_results)

    if completed == 0:
        return {
            "reliability_score": 0.0,
            "reliability_tier": "Low",
            "session_quality_score": 0.0,
            "session_quality_tier": "Low",
            "is_conclusive": False,
            "summary": "No passage data recorded.",
            "audit_trail": ["No passage data recorded."],
        }

    # Extract quality scores per passage
    passage_qualities = [r.get("quality_score", 85.0) for r in passage_results]
    avg_quality = float(np.mean(passage_qualities))
    min_quality = float(np.min(passage_qualities))

    # Base score starts from session execution quality
    score = avg_quality * 0.40  # 40% weight on capture quality
    audit = []

    # 1. Completeness (20% weight)
    completeness_ratio = completed / total_expected
    score += completeness_ratio * 20.0
    if completeness_ratio < 1.0:
        audit.append(f"Assessment incomplete: completed {completed} of {total_expected} stages.")
    else:
        audit.append("Assessment complete: all 5 stages recorded.")

    # 2. Baseline Stability (Passage 1 calibration - 15% weight)
    calib = passage_results[0]
    calib_n = calib.get("n_keystrokes", 0)
    calib_cv = calib.get("inter_key_cv", 0.35)
    
    if calib_n >= 80 and calib_cv <= 0.65:
        score += 15.0
        audit.append("Baseline calibration established with high stability and sufficient sample volume.")
    elif calib_n >= 45:
        score += 10.0
        audit.append("Baseline calibration established with moderate rhythm variation.")
    else:
        score += 4.0
        audit.append("Baseline calibration was short or exhibited elevated variance.")

    # 3. Task Condition Consistency (15% weight)
    # Neutral tasks: P2 (idx 1) and P4 (idx 3)
    # Load tasks: P3 (idx 2) and P5 (idx 4)
    if completed >= 4:
        neutral_scores = [passage_results[1]["strain_score"]]
        if completed >= 4:
            neutral_scores.append(passage_results[3]["strain_score"])
        neutral_spread = abs(neutral_scores[-1] - neutral_scores[0]) if len(neutral_scores) > 1 else 0

        load_scores = [passage_results[2]["strain_score"]]
        if completed >= 5:
            load_scores.append(passage_results[4]["strain_score"])
        load_spread = abs(load_scores[-1] - load_scores[0]) if len(load_scores) > 1 else 0

        avg_spread = (neutral_spread + load_spread) / 2.0
        if avg_spread <= 15.0:
            score += 15.0
            audit.append(f"Strong condition consistency: Neutral and Load task pairs aligned closely (avg spread: {avg_spread:.1f} pts).")
        elif avg_spread <= 30.0:
            score += 10.0
            audit.append(f"Moderate condition consistency: Neutral and Load task pairs showed expected variation (avg spread: {avg_spread:.1f} pts).")
        else:
            score += 5.0
            audit.append(f"Divergent task scores: task pairs exhibited noticeable divergence (avg spread: {avg_spread:.1f} pts).")
    else:
        score += 8.0

    # 4. Statistical & Model Convergence (10% weight)
    # Check agreement between statistical hesitation spikes and model strain score
    diagnostic_results = passage_results[1:]
    stat_agreements = 0
    for r in diagnostic_results:
        prob = r.get("strain_probability", 0.5)
        spikes = r.get("hesitation_spikes", 0)
        cv = r.get("inter_key_cv", 0.3)
        # Agreement if model predicts load and spikes/CV are elevated, or both calm
        if (prob >= 0.5 and (spikes >= 2 or cv >= 0.45)) or (prob < 0.5 and spikes <= 1 and cv < 0.45):
            stat_agreements += 1

    agreement_pct = stat_agreements / max(len(diagnostic_results), 1)
    score += agreement_pct * 10.0
    audit.append(f"Statistical indicator convergence: {agreement_pct*100:.0f}% concordance between ML model and rhythm metrics.")

    final_reliability = round(max(0.0, min(100.0, score)), 1)
    final_quality = round(avg_quality, 1)

    if final_reliability >= 75.0:
        reliability_tier = "High"
    elif final_reliability >= 50.0:
        reliability_tier = "Moderate"
    else:
        reliability_tier = "Low"

    if final_quality >= 75.0:
        quality_tier = "High"
    elif final_quality >= 50.0:
        quality_tier = "Moderate"
    else:
        quality_tier = "Low"

    # Guardrail: Inconclusive Session check
    is_conclusive = True
    inconclusive_reasons = []

    if final_reliability < 40.0:
        is_conclusive = False
        inconclusive_reasons.append("Overall measurement reliability is below the required 40% threshold.")

    if min_quality < 45.0:
        is_conclusive = False
        inconclusive_reasons.append("One or more assessment passages experienced substantial focus disruption or paste events.")

    if completed < 3:
        is_conclusive = False
        inconclusive_reasons.append(f"Only {completed} of {total_expected} passages were submitted; insufficient sample depth.")

    return {
        "reliability_score": final_reliability,
        "reliability_tier": reliability_tier,
        "session_quality_score": final_quality,
        "session_quality_tier": quality_tier,
        "is_conclusive": is_conclusive,
        "inconclusive_reasons": inconclusive_reasons,
        "audit_trail": audit,
        "passage_qualities": passage_qualities,
    }
