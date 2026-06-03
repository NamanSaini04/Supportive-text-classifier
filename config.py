"""Central config: label list, paths, thresholds, crisis resources.

Keeping these in one place means the model, the app, and the tests all agree
on what the categories are and where artifacts live.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# --- Paths ---
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
LOG_PATH = DATA_PROCESSED / "journal_log.csv"

# --- Labels ---
# The six emotional TONES the model predicts. Note: "urgent distress" is NOT
# here on purpose — it is a separate safety signal (see src/distress.py).
TONE_LABELS = ["neutral", "stress", "anxiety", "sadness", "anger", "loneliness"]

# --- Model ---
DEFAULT_MODEL = "logreg"          # logreg gives predict_proba -> confidence scores
RANDOM_STATE = 42

# --- Crisis resources (VERIFY & LOCALIZE before publishing) ---
CRISIS_MESSAGE = (
    "It sounds like you may be going through something really serious. "
    "Please reach out to people who can help right now:\n\n"
    "- **US:** Call or text **988** (Suicide & Crisis Lifeline)\n"
    "- **India:** iCall **+91 9152987821** / AASRA **+91 9820466726**\n"
    "- Or contact a trusted person or your local emergency services.\n\n"
    "You deserve support from a real person."
)

DISCLAIMER = (
    "This tool reflects the *tone* of your words. It is **not** a medical or "
    "mental-health diagnosis and is not a substitute for professional help."
)
