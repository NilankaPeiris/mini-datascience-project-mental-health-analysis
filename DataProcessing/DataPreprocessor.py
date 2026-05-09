"""
01_data_preprocessing.py

Purpose:
- Load the raw Teen Mental Health dataset.
- Check missing values and duplicates.
- Clean text/categorical columns.
- Save a cleaned CSV file and a preprocessing summary.

Run:
    python scripts/01_data_preprocessing.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

# Allow importing config.py when script is run from project root
sys.path.append(str(Path(__file__).resolve().parent))
from config import PROCESSED_DIR, TARGET_COLUMN


class DataPreprocessor:
    """Preprocess raw Teen Mental Health dataset."""
    
    def __init__(self, raw_data_path, processed_dir=None, target_column=None):
        """
        Initialize the preprocessor.
        
        Args:
            raw_data_path (str or Path): Path to the raw CSV data file
            processed_dir (str or Path): Directory to save processed files. Defaults to config.PROCESSED_DIR
            target_column (str): Name of target column. Defaults to config.TARGET_COLUMN
        """
        self.raw_data_path = Path(raw_data_path)
        self.processed_dir = Path(processed_dir) if processed_dir else PROCESSED_DIR
        self.target_column = target_column or TARGET_COLUMN
        self.df = None
        self.summary = {}
    
    def preprocess(self):
        """Run full preprocessing pipeline."""
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.raw_data_path.exists():
            raise FileNotFoundError(
                f"Raw data file not found at {self.raw_data_path}"
            )
        
        # Load data
        df_raw = pd.read_csv(self.raw_data_path)
        original_shape = df_raw.shape
        
        # Clean data
        self.df = self._clean_column_names(df_raw)
        self.df = self._clean_categorical_values(self.df)
        
        # Remove duplicates
        duplicate_count = int(self.df.duplicated().sum())
        self.df = self.df.drop_duplicates().reset_index(drop=True)
        
        # Handle missing values
        missing_before = self.df.isna().sum().to_dict()
        self._handle_missing_values()
        
        # Validate target column
        if self.target_column not in self.df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in dataset.")
        
        # Get column types
        numeric_cols = self.df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        categorical_cols = self.df.select_dtypes(include="object").columns.tolist()
        
        # Save cleaned data
        cleaned_path = self.processed_dir / "teen_mental_health_cleaned.csv"
        self.df.to_csv(cleaned_path, index=False)
        
        # Create summary
        self.summary = {
            "original_shape": original_shape,
            "cleaned_shape": self.df.shape,
            "duplicate_rows_removed": duplicate_count,
            "missing_values_before_imputation": missing_before,
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "target_column": self.target_column,
            "target_distribution": self.df[self.target_column].value_counts().to_dict(),
            "cleaned_data_path": str(cleaned_path),
        }
        
        # Save summary
        summary_path = self.processed_dir / "preprocessing_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(self.summary, f, indent=4)
        
        print("Preprocessing completed.")
        print(f"Cleaned data saved to: {cleaned_path}")
        print(f"Summary saved to: {summary_path}")
        
        return cleaned_path, summary_path
    
    @staticmethod
    def _clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names."""
        df = df.copy()
        df.columns = (
            df.columns
            .str.strip()
            .str.lower()
            .str.replace(" ", "_", regex=False)
            .str.replace("-", "_", regex=False)
        )
        return df
    
    @staticmethod
    def _clean_categorical_values(df: pd.DataFrame) -> pd.DataFrame:
        """Clean object columns by trimming spaces and converting to lowercase."""
        df = df.copy()
        object_cols = df.select_dtypes(include="object").columns
        for col in object_cols:
            df[col] = df[col].astype(str).str.strip().str.lower()
        return df
    
    def _handle_missing_values(self):
        """Fill missing values with median (numeric) or mode (categorical)."""
        numeric_cols = self.df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        categorical_cols = self.df.select_dtypes(include="object").columns.tolist()
        
        for col in numeric_cols:
            if self.df[col].isna().any():
                self.df[col] = self.df[col].fillna(self.df[col].median())
        
        for col in categorical_cols:
            if self.df[col].isna().any():
                mode_value = self.df[col].mode(dropna=True)
                fill_value = mode_value.iloc[0] if not mode_value.empty else "unknown"
                self.df[col] = self.df[col].fillna(fill_value)


def main() -> None:
    """Run preprocessing with default config paths."""
    from config import RAW_DATA_PATH
    
    preprocessor = DataPreprocessor(raw_data_path=RAW_DATA_PATH)
    preprocessor.preprocess()


if __name__ == "__main__":
    main()
