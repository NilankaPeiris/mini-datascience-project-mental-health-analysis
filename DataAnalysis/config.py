"""
Configuration file for Seminar Task 3 project.
Change the paths below when running this project on your own machine.
"""
from pathlib import Path

# ---- Input data ----
RAW_DATA_PATH = Path("data/Teen_Mental_Health_Dataset.csv")

# ---- Output folders ----
PROCESSED_DIR = Path("outputs/processed")
EDA_OUTPUT_DIR = Path("outputs/eda")
MODEL_OUTPUT_DIR = Path("outputs/model")

# ---- Target column ----
TARGET_COLUMN = "depression_label"

# ---- Reproducibility ----
RANDOM_STATE = 42
TEST_SIZE = 0.20
