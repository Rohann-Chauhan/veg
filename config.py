
from pathlib import Path

# Dataset location
DATA_DIR = Path("data/cloud_dataset")

# Image settings
IMAGE_SIZE = 128
BATCH_SIZE = 32

# Train-validation split
VAL_SIZE = 0.20
RANDOM_STATE = 42

# Training settings
EPOCHS = 10
LEARNING_RATE = 1e-4

# Classes
CLASS_NAMES = ["cloud", "not_cloud"]

CLASS_TO_IDX = {
    "cloud": 0,
    "not_cloud": 1
}