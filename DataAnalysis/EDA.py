"""
EDA.py

Purpose:
- Perform exploratory data analysis focusing on Depression level vs other attributes.
- Treat Depression as a categorical variable.
- Include correlation analysis for numeric variables.
- Generate comprehensive visualizations and statistical summaries.

Run:
    python EDA.py
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency, f_oneway

sys.path.append(str(Path(__file__).resolve().parent))
try:
    from config import PROCESSED_DIR, EDA_OUTPUT_DIR, TARGET_COLUMN
except ImportError:
    PROCESSED_DIR = Path("../data")
    EDA_OUTPUT_DIR = Path("../results/eda")
    TARGET_COLUMN = "depression_label"


class EDA:
    """Exploratory Data Analysis focusing on Depression vs other attributes."""
    
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
        
        # Identify column types (excluding target)
        feature_cols = [col for col in self.df.columns if col != self.target_column]
        self.numeric_cols = self.df[feature_cols].select_dtypes(include=["int64", "float64"]).columns.tolist()
        self.categorical_cols = self.df[feature_cols].select_dtypes(include="object").columns.tolist()
        
        print("=" * 60)
        print("EXPLORATORY DATA ANALYSIS: Depression vs Other Attributes")
        print("=" * 60)
        
        # 1. Target variable distribution
        print("\n1. Depression Label Distribution")
        self._analyze_target_distribution(table_dir, chart_dir)
        
        # 2. Depression vs Numeric Attributes
        print("\n2. Depression vs Numeric Attributes")
        self._analyze_numeric_by_depression(table_dir, chart_dir)
        
        # 3. Depression vs Categorical Attributes
        print("\n3. Depression vs Categorical Attributes")
        self._analyze_categorical_by_depression(table_dir, chart_dir)
        
        # 4. Correlation analysis
        print("\n4. Correlation Analysis (Numeric Variables)")
        self._analyze_correlations(table_dir, chart_dir)
        
        # 5. Feature distribution for depressed teenagers only
        print("\n5. Feature Distribution - Depressed Teenagers Only")
        self._analyze_depressed_only(table_dir, chart_dir)
        
        # 6. Feature distribution for non-depressed teenagers only
        print("\n6. Feature Distribution - Non-Depressed Teenagers Only")
        self._analyze_non_depressed_only(table_dir, chart_dir)
        self._analyze_depressed_only(table_dir, chart_dir)
        
        # Create comprehensive summary
        eda_summary = self._create_summary()
        with open(self.output_dir / "eda_summary.json", "w", encoding="utf-8") as f:
            json.dump(eda_summary, f, indent=4)
        
        print("\n" + "=" * 60)
        print("EDA completed.")
        print(f"EDA outputs saved to: {self.output_dir}")
        print("=" * 60)
        
        return eda_summary
    
    def _analyze_target_distribution(self, table_dir: Path, chart_dir: Path):
        """Analyze depression label distribution."""
        depression_stats = self.df[self.target_column].value_counts(normalize=False)
        depression_stats_pct = self.df[self.target_column].value_counts(normalize=True) * 100
        depression_combined = pd.DataFrame({
            "count": depression_stats,
            "percentage": depression_stats_pct
        })
        depression_combined.to_csv(table_dir / "01_depression_distribution.csv")
        
        # Plot
        self._save_bar_chart(
            self.df[self.target_column].value_counts().sort_index(),
            "Depression Label Distribution",
            "Depression Label",
            "Number of Records",
            chart_dir / "01_depression_distribution.png",
        )
        print(f"   - Depression distribution: {depression_combined.to_dict()}")
    
    def _analyze_numeric_by_depression(self, table_dir: Path, chart_dir: Path):
        """Analyze numeric features by depression status."""
        # Group statistics
        grouped_stats = self.df.groupby(self.target_column)[self.numeric_cols].agg(["mean", "std", "min", "max"])
        grouped_stats.to_csv(table_dir / "02_numeric_by_depression_stats.csv")
        
        # ANOVA test results
        anova_results = []
        for col in self.numeric_cols:
            groups = [group[col].values for name, group in self.df.groupby(self.target_column)]
            f_stat, p_value = f_oneway(*groups)
            anova_results.append({
                "feature": col,
                "f_statistic": f_stat,
                "p_value": p_value,
                "significant": "Yes" if p_value < 0.05 else "No"
            })
        
        anova_df = pd.DataFrame(anova_results)
        anova_df.to_csv(table_dir / "02_anova_test_numeric_vs_depression.csv", index=False)
        print(f"   - ANOVA results saved ({len(anova_df)} numeric features tested)")
        
        # Visualizations
        for col in self.numeric_cols:
            self._save_boxplot_by_target(col, chart_dir / f"02_boxplot_{col}_by_depression.png")
            self._save_violin_plot(col, chart_dir / f"02_violin_{col}_by_depression.png")
    
    def _analyze_categorical_by_depression(self, table_dir: Path, chart_dir: Path):
        """Analyze categorical features by depression status."""
        chi_square_results = []
        
        for col in self.categorical_cols:
            # Cross-tabulation
            crosstab = pd.crosstab(self.df[col], self.df[self.target_column], margins=True)
            crosstab.to_csv(table_dir / f"03_crosstab_{col}_vs_depression.csv")
            
            # Chi-square test
            contingency = pd.crosstab(self.df[col], self.df[self.target_column])
            chi2, p_value, dof, expected = chi2_contingency(contingency)
            chi_square_results.append({
                "feature": col,
                "chi_square_statistic": chi2,
                "p_value": p_value,
                "degrees_of_freedom": dof,
                "significant": "Yes" if p_value < 0.05 else "No"
            })
            
            # Visualization
            self._save_categorical_comparison(col, chart_dir / f"03_categorical_{col}_by_depression.png")
        
        chi_square_df = pd.DataFrame(chi_square_results)
        chi_square_df.to_csv(table_dir / "03_chi_square_test_categorical_vs_depression.csv", index=False)
        print(f"   - Chi-square results saved ({len(chi_square_df)} categorical features tested)")
    
    def _analyze_correlations(self, table_dir: Path, chart_dir: Path):
        """Analyze correlations among numeric variables."""
        corr = self.df[self.numeric_cols].corr()
        corr.to_csv(table_dir / "04_correlation_matrix.csv")
        self._save_correlation_heatmap(corr, chart_dir / "04_correlation_heatmap.png")
        print(f"   - Correlation matrix saved")
    
    def _analyze_depressed_only(self, table_dir: Path, chart_dir: Path):
        """Analyze feature distributions for depressed teenagers only."""
        # Filter for depressed teenagers (assuming positive label is 1 or 'depressed')
        depressed_categories = self.df[self.target_column].cat.categories
        depressed_label = depressed_categories[-1] if len(depressed_categories) > 0 else None
        
        df_depressed = self.df[self.df[self.target_column] == depressed_label]
        
        if len(df_depressed) == 0:
            print(f"   - No depressed teenagers found")
            return
        
        print(f"   - Found {len(df_depressed)} depressed teenagers")
        
        # Create subdirectory for depressed-only analysis
        depressed_chart_dir = chart_dir / "depressed_only"
        depressed_chart_dir.mkdir(parents=True, exist_ok=True)
        
        # Summary statistics for depressed teenagers
        depressed_numeric_summary = df_depressed[self.numeric_cols].describe().T
        depressed_numeric_summary.to_csv(table_dir / "05_depressed_numeric_summary.csv")
        
        # Distribution plots for numeric features (depressed only)
        for col in self.numeric_cols:
            self._save_histogram_depressed_only(df_depressed, col, depressed_chart_dir / f"05_hist_{col}_depressed.png")
            self._save_kde_plot(df_depressed, col, depressed_chart_dir / f"05_kde_{col}_depressed.png")
        
        # Distribution plots for categorical features (depressed only)
        for col in self.categorical_cols:
            self._save_categorical_depressed_only(df_depressed, col, depressed_chart_dir / f"05_bar_{col}_depressed.png")
        
        # Correlation among numeric features for depressed teenagers
        corr_depressed = df_depressed[self.numeric_cols].corr()
        corr_depressed.to_csv(table_dir / "05_correlation_matrix_depressed_only.csv")
        self._save_correlation_heatmap(corr_depressed, depressed_chart_dir / "05_correlation_heatmap_depressed.png")
        
        print(f"   - Distribution plots saved for depressed teenagers")
    
    def _analyze_non_depressed_only(self, table_dir: Path, chart_dir: Path):
        """Analyze feature distributions for non-depressed teenagers only."""
        # Filter for non-depressed teenagers (assuming first category is non-depressed)
        depressed_categories = self.df[self.target_column].cat.categories
        non_depressed_label = depressed_categories[0] if len(depressed_categories) > 0 else None
        
        df_non_depressed = self.df[self.df[self.target_column] == non_depressed_label]
        
        if len(df_non_depressed) == 0:
            print(f"   - No non-depressed teenagers found")
            return
        
        print(f"   - Found {len(df_non_depressed)} non-depressed teenagers")
        
        # Create subdirectory for non-depressed-only analysis
        non_depressed_chart_dir = chart_dir / "non_depressed_only"
        non_depressed_chart_dir.mkdir(parents=True, exist_ok=True)
        
        # Summary statistics for non-depressed teenagers
        non_depressed_numeric_summary = df_non_depressed[self.numeric_cols].describe().T
        non_depressed_numeric_summary.to_csv(table_dir / "06_non_depressed_numeric_summary.csv")
        
        # Distribution plots for numeric features (non-depressed only)
        for col in self.numeric_cols:
            self._save_histogram_non_depressed_only(df_non_depressed, col, non_depressed_chart_dir / f"06_hist_{col}_non_depressed.png")
            self._save_kde_plot_non_depressed_only(df_non_depressed, col, non_depressed_chart_dir / f"06_kde_{col}_non_depressed.png")
        
        # Distribution plots for categorical features (non-depressed only)
        for col in self.categorical_cols:
            self._save_categorical_non_depressed_only(df_non_depressed, col, non_depressed_chart_dir / f"06_bar_{col}_non_depressed.png")
        
        # Correlation among numeric features for non-depressed teenagers
        corr_non_depressed = df_non_depressed[self.numeric_cols].corr()
        corr_non_depressed.to_csv(table_dir / "06_correlation_matrix_non_depressed_only.csv")
        self._save_correlation_heatmap_non_depressed(corr_non_depressed, non_depressed_chart_dir / "06_correlation_heatmap_non_depressed.png")
        
        print(f"   - Distribution plots saved for non-depressed teenagers")
    
    def _create_summary(self) -> dict:
        """Create comprehensive EDA summary."""
        return {
            "total_rows": int(self.df.shape[0]),
            "total_columns": int(self.df.shape[1]),
            "numeric_features": self.numeric_cols,
            "categorical_features": self.categorical_cols,
            "target_column": self.target_column,
            "target_data_type": "categorical",
            "target_categories": self.df[self.target_column].cat.categories.tolist(),
            "target_distribution": self.df[self.target_column].value_counts().to_dict(),
            "target_distribution_pct": (self.df[self.target_column].value_counts(normalize=True) * 100).round(2).to_dict(),
            "analysis_focus": "Depression level vs other attributes (bivariate analysis)",
            "statistical_tests_used": ["ANOVA for numeric features", "Chi-square for categorical features"],
            "note": "Comprehensive analysis of how depression correlates with each feature. Significant features (p < 0.05) are strong predictors.",
        }
    
    @staticmethod
    def _save_bar_chart(series: pd.Series, title: str, xlabel: str, ylabel: str, output_path: Path) -> None:
        """Save a bar chart."""
        plt.figure(figsize=(8, 5))
        series.plot(kind="bar", color="steelblue")
        plt.title(title, fontsize=14, fontweight="bold")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
    
    def _save_boxplot_by_target(self, column: str, output_path: Path) -> None:
        """Save a boxplot comparing numeric column by target variable."""
        plt.figure(figsize=(8, 5))
        self.df.boxplot(column=column, by=self.target_column)
        plt.title(f"{column} by {self.target_column}", fontweight="bold")
        plt.suptitle("")
        plt.xlabel(self.target_column)
        plt.ylabel(column)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
    
    def _save_violin_plot(self, column: str, output_path: Path) -> None:
        """Save a violin plot comparing numeric column by target variable."""
        plt.figure(figsize=(8, 5))
        sns.violinplot(data=self.df, x=self.target_column, y=column, palette="Set2")
        plt.title(f"Distribution of {column} by {self.target_column}", fontweight="bold")
        plt.xlabel(self.target_column)
        plt.ylabel(column)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
    
    def _save_categorical_comparison(self, column: str, output_path: Path) -> None:
        """Save a stacked bar chart for categorical variable vs depression."""
        plt.figure(figsize=(10, 6))
        crosstab = pd.crosstab(self.df[column], self.df[self.target_column], normalize="index") * 100
        crosstab.plot(kind="bar", stacked=False)
        plt.title(f"{column} vs {self.target_column} (% by {column})", fontweight="bold")
        plt.xlabel(column)
        plt.ylabel("Percentage (%)")
        plt.legend(title=self.target_column)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
    
    @staticmethod
    def _save_correlation_heatmap(corr: pd.DataFrame, output_path: Path) -> None:
        """Save a correlation heatmap using seaborn."""
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, cbar_kws={"label": "Correlation"})
        plt.title("Correlation Heatmap - Numeric Variables", fontweight="bold")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
    
    @staticmethod
    def _save_histogram_depressed_only(df: pd.DataFrame, column: str, output_path: Path) -> None:
        """Save a histogram for depressed teenagers only."""
        plt.figure(figsize=(8, 5))
        plt.hist(df[column], bins=20, color="crimson", edgecolor="black", alpha=0.7)
        plt.title(f"Distribution of {column} (Depressed Teenagers)", fontweight="bold")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
    
    @staticmethod
    def _save_kde_plot(df: pd.DataFrame, column: str, output_path: Path) -> None:
        """Save a KDE (kernel density estimate) plot for depressed teenagers only."""
        plt.figure(figsize=(8, 5))
        df[column].plot(kind="density", color="crimson", linewidth=2)
        plt.title(f"KDE Plot of {column} (Depressed Teenagers)", fontweight="bold")
        plt.xlabel(column)
        plt.ylabel("Density")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
    
    @staticmethod
    def _save_categorical_depressed_only(df: pd.DataFrame, column: str, output_path: Path) -> None:
        """Save a bar chart for categorical feature (depressed teenagers only)."""
        plt.figure(figsize=(8, 5))
        value_counts = df[column].value_counts()
        value_counts.plot(kind="bar", color="crimson", edgecolor="black", alpha=0.7)
        plt.title(f"Distribution of {column} (Depressed Teenagers)", fontweight="bold")
        plt.xlabel(column)
        plt.ylabel("Count")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
    
    @staticmethod
    def _save_histogram_non_depressed_only(df: pd.DataFrame, column: str, output_path: Path) -> None:
        """Save a histogram for non-depressed teenagers only."""
        plt.figure(figsize=(8, 5))
        plt.hist(df[column], bins=20, color="seagreen", edgecolor="black", alpha=0.7)
        plt.title(f"Distribution of {column} (Non-Depressed Teenagers)", fontweight="bold")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
    
    @staticmethod
    def _save_kde_plot_non_depressed_only(df: pd.DataFrame, column: str, output_path: Path) -> None:
        """Save a KDE (kernel density estimate) plot for non-depressed teenagers only."""
        plt.figure(figsize=(8, 5))
        df[column].plot(kind="density", color="seagreen", linewidth=2)
        plt.title(f"KDE Plot of {column} (Non-Depressed Teenagers)", fontweight="bold")
        plt.xlabel(column)
        plt.ylabel("Density")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
    
    @staticmethod
    def _save_categorical_non_depressed_only(df: pd.DataFrame, column: str, output_path: Path) -> None:
        """Save a bar chart for categorical feature (non-depressed teenagers only)."""
        plt.figure(figsize=(8, 5))
        value_counts = df[column].value_counts()
        value_counts.plot(kind="bar", color="seagreen", edgecolor="black", alpha=0.7)
        plt.title(f"Distribution of {column} (Non-Depressed Teenagers)", fontweight="bold")
        plt.xlabel(column)
        plt.ylabel("Count")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
    
    @staticmethod
    def _save_correlation_heatmap_non_depressed(corr: pd.DataFrame, output_path: Path) -> None:
        """Save a correlation heatmap for non-depressed teenagers using seaborn."""
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn", center=0, cbar_kws={"label": "Correlation"})
        plt.title("Correlation Heatmap - Non-Depressed Teenagers", fontweight="bold")
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
