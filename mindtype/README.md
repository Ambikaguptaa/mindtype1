# MindType — Typing-Based Cognitive Strain Analysis

> **Primary Project Statement:**  
> **"MindType analyzes keystroke dynamics to identify changes in typing behavior associated with cognitive or task-related strain."**

MindType is a research prototype that examines micro-timing, rhythm variability, pause frequency, and correction behaviors from live typing to detect measurable behavioral shifts associated with cognitive load and task difficulty.

---

## ⚠️ Important Scientific Positioning & Limitations

* **Behavioral Estimate, Not a Medical Diagnosis:** MindType analyzes keystroke dynamics and physical typing mechanics. It does **not** diagnose depression, anxiety, ADHD, psychiatric disorders, mental illness, or any clinical condition.
* **Non-Clinical Terminology:** Outputs are structured as a **Behavioral Strain Index (0–100)** and **Measurement Reliability Score (0–100)**.
* **Alternative Factors:** Typing variations can arise from numerous everyday non-emotional causes, including physical fatigue, environmental distractions, keyboard hardware differences, unfamiliar text, task complexity, or temporary pacing shifts.
* **Synthetic Development Validation:** The underlying machine learning model in this repository was trained and evaluated on synthetic developmental data. High accuracy on synthetic benchmark data demonstrates algorithmic separation under controlled conditions; it does **not** represent real-world clinical accuracy.

| Validation Tier | Current Status | Definition |
|---|---|---|
| **Synthetic Development Validation** | **Complete** | Model separation verified on simulated cognitive load distributions (development prototype). |
| **Real-World Empirical Validation** | **Roadmap Phase 1** | Testing on diverse hardware and unconstrained typing tasks with consented participants. |
| **Clinical / Psychological Validation** | **Roadmap Phase 2** | Correlating keystroke dynamics against validated self-report instruments (NASA-TLX, PSS-10). |

---

## 🧭 End-to-End User Journey

```
Landing Page (Privacy & Methodology)
  ↓
Pre-Assessment Focus Screen (Focus Instructions & Mandatory Readiness Check)
  ↓
5-Stage Alternating Assessment:
  • Passage 1: Natural Typing Baseline (Calibration)
  • Passage 2: Neutral Typing Task 1
  • Passage 3: Cognitive Load Task 1 (Phonetic & Motor Complexity)
  • Passage 4: Neutral Typing Task 2 (Recovery Check)
  • Passage 5: Cognitive Load Task 2 (Dual-Task & Mental Sequencing)
  ↓
Session Quality & Inactivity Validation (Focus, Blur, Paste Checks)
  ↓
Feature Extraction & Robust Statistics (Median, MAD, CV, Percentiles, Bursts)
  ↓
Personal Baseline Comparison (% Shifts vs Passage 1 Calibration)
  ↓
Machine Learning Inference & Explainability Attribution
  ↓
Measurement Reliability Scoring & Inconclusive Guardrails
  ↓
Comprehensive Research Report & Multi-Format Exports (HTML & TXT)
```

---

## 🔒 Privacy-First Architecture

1. **Zero Raw Text Storage:** Typed message content is **never retained, stored, or analyzed**. Only mechanical keystroke metadata (inter-key timing deltas, dwell intervals, binary backspace flags, error counts) is processed.
2. **Cryptographic Pseudonymization:** User identifiers are transformed via salted SHA-256 one-way hashing prior to baseline association.
3. **Encrypted Baselines:** Baseline feature stores are encrypted at rest using AES-128-CBC via Fernet.
4. **Data Minimization:** No clipboard contents, microphone, camera, or unrelated browser data are accessed. Paste events are blocked to preserve keystroke timing validity.
5. **Differential Privacy:** Mathematical Laplace noise mechanism available for aggregate/cohort analytical queries, strictly excluded from altering individual live reports.

---

## 📁 Project Structure

```
mindtype/
├── requirements.txt            # Python dependencies (TensorFlow, Streamlit, Plotly, Scikit-learn, etc.)
├── README.md                   # Research documentation and roadmap
├── data/
│   ├── raw_keystrokes.csv      # Synthetic developmental training dataset
│   └── session_log.csv         # Local session audit log
├── models/
│   ├── lstm_model.keras        # Trained sequential LSTM model
│   ├── scaler.npz              # Standardization parameters
│   ├── baselines.enc           # Encrypted baseline storage
│   ├── baselines_plain.csv     # Local development baselines
│   └── metrics.json            # Development evaluation metrics
└── src/
    ├── dashboard.py            # Streamlit dashboard with Plus Jakarta Sans styling
    ├── paragraphs.py           # 5-stage alternating assessment protocol (symbol-free)
    ├── keystroke_component/    # In-browser keystroke capture widget (HTML/JS)
    │   └── index.html          # Tab visibility, blur, paste prevention, pause tracking
    ├── keystroke_widget.py     # Python component wrapper
    ├── features.py             # Feature engineering with robust statistics (MAD, CV, percentiles)
    ├── reliability.py          # Session Quality & Measurement Reliability Engine
    ├── interpretation.py       # Behavioral interpretation layer & cautious findings
    ├── models_engine.py        # Multi-model benchmarks & explainability attribution
    ├── visuals.py              # Interactive Plotly charts (radar, timeline, baseline deltas)
    ├── report.py               # 5-passage report synthesis & HTML/TXT generator
    ├── infer_live.py           # Live scoring pipeline & statistical tests
    ├── dsp.py                  # Privacy, pseudonymization, and encryption controls
    ├── generate_data.py        # Synthetic dataset generator
    ├── model.py                # LSTM architecture definition
    └── train.py                # End-to-end training pipeline
```

---

## 🚀 How to Run on Localhost

### Prerequisites
* Python 3.10 to 3.12 (TensorFlow pre-built wheels support up to Python 3.12).
* PowerShell or standard terminal.

### Quick Start
```powershell
# 1. Navigate to the project directory
cd "c:\Users\ambik\Downloads\mindtype-project (1) 1\mindtype"

# 2. If not already active, use the Python 3.12 virtual environment:
.\.venv\Scripts\Activate.ps1

# 3. Launch the Streamlit dashboard
streamlit run src/dashboard.py
```

Open your browser to:  
👉 **[http://localhost:8501](http://localhost:8501)**

---

## 🔬 Real-Data Research Roadmap

MindType's long-term scientific roadmap transitions from synthetic development data to empirically grounded human research:

```
Consented Participant Cohort
             ↓
Continuous Keystroke Dynamics Capture (Natural & Controlled Tasks)
             +
Validated Self-Report Measurements (NASA-TLX, PSS-10, PANAS, Sleep/Fatigue Logs)
             ↓
Empirical Behavioral Ground Truth Labels
             ↓
Robust Feature Engineering & Selection (Time-series characteristics via tsfresh)
             ↓
Cross-Validation & Model Comparison (LSTM, XGBoost, Random Forest)
             ↓
External Multi-Device & Naturalistic Validation
             ↓
Peer-Reviewed Research Dissemination
```

### Prospective Self-Report Dimensions:
* **Task Load Index (NASA-TLX):** Mental demand, temporal demand, and perceived effort.
* **Perceived Stress Scale (PSS-10):** Longitudinal stress variability.
* **State-Trait Inventory (STAI):** Situational tension vs resting disposition.
* **Fatigue & Sleep Quality Indices:** Distinguishing physical tiredness from cognitive load.
