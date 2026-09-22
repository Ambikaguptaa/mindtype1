"""
dsp.py  (DSP — Data Security & Privacy)
-----------------------------------------
Production-grade privacy and security implementations for MindType:
  1. Pseudonymization: Salted SHA-256 one-way cryptographic hashing of identifiers.
  2. Data Minimization: Zero raw typed message content is captured or stored;
     only keystroke timing deltas and binary error/backspace flags are retained.
  3. Encryption-at-Rest: Fernet (AES-128-CBC + HMAC-SHA256) encryption for
     stored baseline feature stores.
  4. Differential Privacy: Mathematical Laplace mechanism for aggregate/cohort
     analytical queries (e.g. population mean typing speed), deliberately
     NOT added arbitrarily to individual single-session reports.
"""
import hashlib
import os
import secrets
import numpy as np
from cryptography.fernet import Fernet

SECRET_SALT = os.environ.get("MINDTYPE_SALT", "mindtype-research-prototype-salt-v2")


def pseudonymize_id(user_id: str) -> str:
    """One-way salted cryptographic hash: the real user_id is never stored
    alongside behavioral typing features. Irreversible without the salt and original ID.
    """
    if not user_id:
        return f"u_{secrets.token_hex(8)}"
    uid_str = str(user_id).strip()
    if not uid_str:
        return f"u_{secrets.token_hex(8)}"
    if uid_str.startswith("u_") and len(uid_str) == 18 and all(c in "0123456789abcdefABCDEF" for c in uid_str[2:]):
        return uid_str
    h = hashlib.sha256((SECRET_SALT + uid_str).encode()).hexdigest()
    return f"u_{h[:16]}"


def pseudonymize_user_id(user_id: str = None) -> str:
    """One-way salted cryptographic hash for participant labels and identifiers.
    Exposed function adhering to MindType DSP architecture. Guarantees deterministic,
    consistent hashing with the configured SECRET_SALT while safely handling
    edge cases (missing, None, empty, already pseudonymized).
    """
    if user_id is None:
        return "Unavailable"
    uid_str = str(user_id).strip()
    if not uid_str or uid_str.lower() in ("none", "unavailable", "unknown"):
        return "Unavailable"
    if uid_str.startswith("u_") and len(uid_str) == 18 and all(c in "0123456789abcdefABCDEF" for c in uid_str[2:]):
        return uid_str
    h = hashlib.sha256((SECRET_SALT + uid_str).encode()).hexdigest()
    return f"u_{h[:16]}"


def get_or_create_key(path: str = "models/dsp.key") -> bytes:
    """Retrieves existing encryption key or securely generates a new one.
    Prefers environment variable MINDTYPE_ENCRYPTION_KEY if present.
    """
    env_key = os.environ.get("MINDTYPE_ENCRYPTION_KEY")
    if env_key:
        return env_key.encode()

    if os.path.exists(path):
        with open(path, "rb") as f:
            return f.read().strip()

    key = Fernet.generate_key()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(key)
    return key


def encrypt_bytes(data: bytes, key: bytes = None) -> bytes:
    """Encrypts payload bytes using AES-128-CBC via Fernet."""
    if key is None:
        key = get_or_create_key()
    return Fernet(key).encrypt(data)


def decrypt_bytes(token: bytes, key: bytes = None) -> bytes:
    """Decrypts Fernet ciphertext payload."""
    if key is None:
        key = get_or_create_key()
    return Fernet(key).decrypt(token)


def add_laplace_noise(value: float, epsilon: float, sensitivity: float) -> float:
    """Mathematical Laplace mechanism for Differential Privacy in aggregate queries.
    
    Used strictly for cohort / population queries (e.g. average typing speed
    across 50 participants) where individual data points must remain k-anonymous
    and mathematically protected against membership inference attacks.
    
    NEVER injected into an individual user's personal report.
    """
    scale = sensitivity / max(epsilon, 1e-4)
    noise = float(np.random.laplace(0, scale))
    return value + noise
