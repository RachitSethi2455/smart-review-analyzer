"""Project paths and shared settings."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
SAMPLE_CSV = ROOT / "data" / "sample_reviews.csv"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

for _d in (DATA_RAW, DATA_PROCESSED, MODELS_DIR, RESULTS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

SPACY_MODEL = "en_core_web_sm"
BEST_MODEL_PATH = MODELS_DIR / "sentiment_best.joblib"
TRANSFORMER_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"

# VADER's recommended cut-offs on the compound score
VADER_POS = 0.05
VADER_NEG = -0.05

RANDOM_STATE = 42
