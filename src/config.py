from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_ROOT = BASE_DIR / "data"
DEVICE = "cuda"

NUM_SLICES = 8
IMAGE_SIZE = 224
