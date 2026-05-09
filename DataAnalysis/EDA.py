"""
02_eda.py

Purpose:
- Perform exploratory data analysis for the Teen Mental Health dataset.
- Save summary tables and charts into a configurable output folder.

Run:
    python scripts/02_eda.py
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from config import PROCESSED_DIR, EDA_OUTPUT_DIR, TARGET_COLUMN


class EDA:
    """Exploratory Data Analysis for Teen Mental Health dataset."""
    
    def __init__(self, dataset_path, output_dir=None, target_column=None):
        """
        Initialize the EDA analyzer.
        
        Args:
            dataset_path (str or Path): Path to the cleaned CSV dataset
            output_dir (str or Path): Directory to save EDA outputs. Defaults to config.EDA_OUTPUT_DIR
            target_column (str): Name of target column. Defaults to config.TARGET_COLUMN
        """
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir) if output_dir else EDA_OUTPUT_DIR
        self.target_column = target_column or TARGET_COLUMN
        self.df = None
        self.numeric_cols = []
        self.categorical_cols = []
    
    def analyze(self):
        """Run full EDA pipeline and save outputs."""
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        chart_dir = self.output_dir / "charts"
        table_dir = self.output_dir / "tables"
        chart_dir.mkdir(parents=True, exist_ok=True)
        table_dir.mkdir(parents=True, exist_ok=True)
        
        # Load dataset
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found at {self.dataset_path}")
        
        self.df = pd.read_csv(self.dataset_path)
        
        # Convert target column to categorical
        self.df[self.target_column] = self.df[self.target_column].astype("category")
        
        # Identify column types
        self.numeric_cols = self.df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        self.categorical_cols = self.df.select_dtypes(include="object").columns.tolist()
        
        # Generate summary tables
        self.df.describe().T.to_csv(table_dir / "numeric_summary.csv")
        self.df[self.categorical_cols].describe().T.to_csv(table_dir / "categorical_summary.csv")
        
        # Depression label statistics
        depression_stats = self.df[self.target_column].value_counts(normalize=False)
        depression_stats_pct = self.df[self.target_column].value_counts(normalize=True) * 100
        depression_combined = pd.DataFrame({
            "count": depression_stats,
            "percentage": depression_stats_pct
        })
        depression_combined.to_csv(table_dir / "target_distribution.csv")
        
        # Generate charts
        self._save_bar_chart(
            self.df[self.target_column].value_counts().sort_index(),
            "Depression Label Distribution",
            "Depression Label",
            "Number of Records",
            chart_dir / "target_distribution.png",
        )
        
        # Univariate analysis for numeric variables
        for col in self.numeric_cols:
            self._save_histogram(col, chart_dir / f"hist_{col}.png")
        
        # Categorical count charts
        for col in self.categorical_cols:
            self._save_bar_chart(
                self.df[col].value_counts(),
                f"Distribution of {col}",
                col,
                "Count",
                chart_dir / f"bar_{col}.png",
            )
        
        # Cross-tabulation analysis
        for col in self.categorical_cols:
            crosstab = pd.crosstab(self.df[col], self.df[self.target_column], margins=True)
            crosstab.to_csv(table_dir / f"crosstab_{col}_vs_depression.csv")
        
        # Bivariate analysis: numeric vs target
        for col in self.numeric_cols:
            if col != self.target_column:
                self._save_boxplot_by_target(col, chart_dir / f"box_{col}_by_depression.png")
        
        # Correlation analysis
        corr = self.df[self.numeric_cols].corr()
        corr.to_csv(table_dir / "correlation_matrix.csv")
        self._save_correlation_heatmap(corr, chart_dir / "correlation_heatmap.png")
        
        # Grouped statistics by depression label
        grouped_means = self.df.groupby(self.target_column)[self.numeric_cols].mean()
        grouped_means.to_csv(table_dir / "numeric_means_by_depression_label.csv")
        
        # Create summary JSON
        eda_summary = {
            "rows": int(self.df.shape[0]),
            "columns": int(self.df.shape[1]),
            "numeric_columns": self.numeric_cols,
            "categorical_columns": self.categorical_cols,
            "target_column": self.target_column,
            "target_data_type": "categorical",
            "target_categories": self.df[self.target_column].cat.categories.tolist(),
            "target_distribution": self.df[self.target_column].value_counts().to_dict(),
            "target_distribution_pct": (self.df[self.target_column].value_counts(normalize=True) * 100).to_dict(),
            "note": "Because the depressed class is very small, model evaluation should focus on recall, F1, ROC-AUC, and PR-AUC instead of accuracy alone.",
        }
        
        with open(self.output_dir / "eda_summary.json", "w", encoding="utf-8") as f:
            json.dump(eda_summary, f, indent=4)
        
        print("EDA completed.")
        print(f"EDA outputs saved to: {self.output_dir}")
        
        return eda_summary
    
    @staticmethod
    def _save_bar_chart(series: pd.Series, title: str, xlabel: str, ylabel: str, output_path: Path) -> None:
        """Save a bar chart."""
        plt.figure(figsize=(8, 5))
        series.plot(kind="bar")
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
    
    def _save_histogram(self, column: str, output_path: Path) -> None:
        """Save a histogram for a numeric column."""
        plt.figure(figsize=(8, 5))
        plt.hist(self.df[column], bins=20, edgecolor="black")
        plt.title(f"Distribution of {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
    
    def _save_boxplot_by_target(self, column: str, output_path: Path) -> None:
        """Save a boxplot comparing numeric column by target variable."""
        plt.figure(figsize=(8, 5))
        self.df.boxplot(column=column, by=self.target_column)
        plt.title(f"{column} by Depression Status")
        plt.suptitle("")
        plt.xlabel("Depression Label")
        plt.ylabel(column)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
    
    @staticmethod
    def _save_correlation_heatmap(corr: pd.DataFrame, output_path: Path) -> None:
        """Save a correlation heatmap."""
        plt.figure(figsize=(10, 8))
        plt.imshow(corr, aspect="auto")
        plt.colorbar(label="Correlation")
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
        plt.yticks(range(len(corr.index)), corr.index)
        plt.title("Correlation Heatmap - Numeric Variables")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()


def main() -> None:
    """Run EDA with default config paths."""
    cleaned_path = PROCESSED_DIR / "teen_mental_health_cleaned.csv"
    eda = EDA(dataset_path=cleaned_path, output_dir=EDA_OUTPUT_DIR, target_column=TARGET_COLUMN)
    eda.analyze()


if __name__ == "__main__":
    main()
