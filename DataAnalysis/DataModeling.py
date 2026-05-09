"""
03_train_model.py

Purpose:
- Train and compare three classification models to predict depression_label.
- Handle categorical encoding and numeric scaling using scikit-learn Pipelines.
- Handle class imbalance using balanced sample weights during model training.
- Report performance metrics for all three models.
- Select the best model using PR-AUC, which is more suitable than accuracy for imbalanced data.
- Save model artifacts, model comparison metrics, confusion matrices, ROC curves, PR curves,
  and feature importance / coefficient artifacts where available.

Run:
    python scripts/03_train_model.py
"""

import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

# Allow importing config.py when script is run from project root.
sys.path.append(str(Path(__file__).resolve().parent))
from config import PROCESSED_DIR, MODEL_OUTPUT_DIR, TARGET_COLUMN, RANDOM_STATE, TEST_SIZE


POSITIVE_LABEL = 1
NEGATIVE_LABEL = 0


class ModelTrainer:
    """Train and evaluate classification models for depression prediction."""
    
    def __init__(self, dataset_path, output_dir=None, target_column=None, test_size=None, random_state=None):
        """
        Initialize the ModelTrainer.
        
        Args:
            dataset_path (str or Path): Path to the cleaned CSV dataset
            output_dir (str or Path): Directory to save model outputs. Defaults to config.MODEL_OUTPUT_DIR
            target_column (str): Name of target column. Defaults to config.TARGET_COLUMN
            test_size (float): Test set size. Defaults to config.TEST_SIZE
            random_state (int): Random seed. Defaults to config.RANDOM_STATE
        """
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir) if output_dir else MODEL_OUTPUT_DIR
        self.target_column = target_column or TARGET_COLUMN
        self.test_size = test_size if test_size is not None else TEST_SIZE
        self.random_state = random_state if random_state is not None else RANDOM_STATE
        
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.fitted_models = {}
        self.comparison_df = None
    
    def train(self):
        """Run the full model training pipeline."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found at {self.dataset_path}")
        
        self.df = pd.read_csv(self.dataset_path)
        
        if self.target_column not in self.df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in dataset.")
        
        # Prepare features and target
        self.X = self.df.drop(columns=[self.target_column])
        self.y = self.df[self.target_column]
        
        # Train-test split with stratification
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=self.y,
        )
        
        # Handle class imbalance
        sample_weights = compute_sample_weight(class_weight="balanced", y=self.y_train)
        
        # Build preprocessor and models
        preprocessor = self._build_preprocessor(self.X)
        models = self._build_models(preprocessor)
        
        # Train and evaluate all models
        all_metrics = []
        for model_name, clf in models.items():
            print(f"Training model: {model_name}")
            clf.fit(self.X_train, self.y_train, model__sample_weight=sample_weights)
            self.fitted_models[model_name] = clf
            
            metrics, _ = self._evaluate_model(
                model_name=model_name,
                clf=clf,
                X_test=self.X_test,
                y_test=self.y_test,
            )
            all_metrics.append(metrics)
            
            # Save fitted model
            model_dir = self.output_dir / model_name
            joblib.dump(clf, model_dir / f"{model_name}_pipeline.joblib")
        
        # Compare models
        self.comparison_df = pd.DataFrame(all_metrics).sort_values(
            by=["pr_auc_average_precision", "f1_score", "recall"],
            ascending=False,
        )
        self.comparison_df.to_csv(self.output_dir / "model_comparison_metrics.csv", index=False)
        self._save_model_comparison_plot(self.comparison_df)
        
        # Save best model
        best_model_name = self.comparison_df.iloc[0]["model_name"]
        best_model = self.fitted_models[best_model_name]
        joblib.dump(best_model, self.output_dir / "best_depression_prediction_model.joblib")
        self._save_best_model_explanation(best_model_name, best_model)
        
        # Save training summary
        class_distribution = self._get_class_distribution()
        summary = {
            "class_imbalance_handling": "Balanced sample weights were calculated from the training target distribution using compute_sample_weight(class_weight='balanced'). These weights were passed to every model during fit, so the minority depressed class receives higher importance during training.",
            "model_selection_approach": "Three models were trained using the same train/test split and same preprocessing pipeline. The best model was selected mainly by PR-AUC because the dataset is highly imbalanced. F1-score and recall were also considered as secondary indicators.",
            "best_model": best_model_name,
            "selection_metric": "pr_auc_average_precision",
            "class_distribution": class_distribution,
            "important_note": "Accuracy can be misleading for this dataset because the depressed class is much smaller than the not depressed class. Discuss precision, recall, F1-score, ROC-AUC, PR-AUC, and confusion matrix in the report.",
        }
        
        with open(self.output_dir / "model_training_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4)
        
        print("Model comparison completed.")
        print(f"Model outputs saved to: {self.output_dir}")
        print("\nModel comparison metrics:")
        print(self.comparison_df.to_string(index=False))
        print(f"\nBest model selected: {best_model_name}")
        
        return self.comparison_df, best_model_name
    
    def _get_class_distribution(self) -> dict:
        """Calculate class distribution statistics."""
        return {
            "total_rows": int(len(self.y)),
            "train_rows": int(len(self.y_train)),
            "test_rows": int(len(self.y_test)),
            "positive_class_total_depressed": int((self.y == POSITIVE_LABEL).sum()),
            "negative_class_total_not_depressed": int((self.y == NEGATIVE_LABEL).sum()),
            "positive_class_train_depressed": int((self.y_train == POSITIVE_LABEL).sum()),
            "negative_class_train_not_depressed": int((self.y_train == NEGATIVE_LABEL).sum()),
            "positive_class_test_depressed": int((self.y_test == POSITIVE_LABEL).sum()),
            "negative_class_test_not_depressed": int((self.y_test == NEGATIVE_LABEL).sum()),
        }
    
    @staticmethod
    def _get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
        """Extract feature names after preprocessing and one-hot encoding."""
        feature_names = []
        for name, transformer, columns in preprocessor.transformers_:
            if name == "remainder" and transformer == "drop":
                continue
            
            if hasattr(transformer, "named_steps"):
                last_step = list(transformer.named_steps.values())[-1]
                if hasattr(last_step, "get_feature_names_out"):
                    feature_names.extend(last_step.get_feature_names_out(columns))
                else:
                    feature_names.extend(columns)
            else:
                feature_names.extend(columns)
        
        return list(feature_names)
    
    @staticmethod
    def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
        """Create preprocessing steps for numeric and categorical columns."""
        numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
        categorical_features = X.select_dtypes(include="object").columns.tolist()
        
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        
        return ColumnTransformer(
            transformers=[
                ("numeric", numeric_pipeline, numeric_features),
                ("categorical", categorical_pipeline, categorical_features),
            ]
        )
    
    def _build_models(self, preprocessor: ColumnTransformer) -> Dict[str, Pipeline]:
        """Create the three candidate classification models."""
        return {
            "logistic_regression": Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=2000,
                            random_state=self.random_state,
                            solver="liblinear",
                        ),
                    ),
                ]
            ),
            "random_forest": Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=300,
                            random_state=self.random_state,
                            min_samples_leaf=2,
                        ),
                    ),
                ]
            ),
            "gradient_boosting": Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    (
                        "model",
                        GradientBoostingClassifier(random_state=self.random_state),
                    ),
                ]
            ),
        }
    
    @staticmethod
    def _save_confusion_matrix_plot(cm, output_path: Path, title: str) -> None:
        """Save a simple confusion matrix plot."""
        plt.figure(figsize=(6, 5))
        plt.imshow(cm)
        plt.title(title)
        plt.colorbar()
        plt.xticks([0, 1], ["Not Depressed", "Depressed"])
        plt.yticks([0, 1], ["Not Depressed", "Depressed"])
        plt.xlabel("Predicted Label")
        plt.ylabel("Actual Label")
        
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, cm[i, j], ha="center", va="center")
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
    
    def _evaluate_model(
        self,
        model_name: str,
        clf: Pipeline,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Tuple[dict, pd.DataFrame]:
        """Evaluate one fitted model and save its model-specific artifacts."""
        model_dir = self.output_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]
        
        metrics = {
            "model_name": model_name,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1_score": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_proba),
            "pr_auc_average_precision": average_precision_score(y_test, y_proba),
        }
        
        report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        report_df = pd.DataFrame(report_dict).T
        report_df.to_csv(model_dir / "classification_report.csv")
        
        cm = confusion_matrix(y_test, y_pred, labels=[NEGATIVE_LABEL, POSITIVE_LABEL])
        pd.DataFrame(
            cm,
            index=["actual_not_depressed", "actual_depressed"],
            columns=["predicted_not_depressed", "predicted_depressed"],
        ).to_csv(model_dir / "confusion_matrix.csv")
        self._save_confusion_matrix_plot(
            cm,
            model_dir / "confusion_matrix.png",
            title=f"Confusion Matrix - {model_name}",
        )
        
        RocCurveDisplay.from_predictions(y_test, y_proba)
        plt.title(f"ROC Curve - {model_name}")
        plt.tight_layout()
        plt.savefig(model_dir / "roc_curve.png", dpi=300)
        plt.close()
        
        PrecisionRecallDisplay.from_predictions(y_test, y_proba)
        plt.title(f"Precision-Recall Curve - {model_name}")
        plt.tight_layout()
        plt.savefig(model_dir / "precision_recall_curve.png", dpi=300)
        plt.close()
        
        with open(model_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4)
        
        return metrics, report_df
    
    def _save_model_comparison_plot(self, comparison_df: pd.DataFrame) -> None:
        """Save a grouped metric comparison chart for all models."""
        metric_cols = ["precision", "recall", "f1_score", "roc_auc", "pr_auc_average_precision"]
        plot_df = comparison_df.set_index("model_name")[metric_cols]
        
        ax = plot_df.plot(kind="bar", figsize=(11, 6))
        ax.set_title("Model Performance Comparison")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.05)
        ax.legend(loc="lower right")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plt.savefig(self.output_dir / "model_comparison_metrics.png", dpi=300)
        plt.close()
    
    def _save_best_model_explanation(self, best_model_name: str, clf: Pipeline) -> None:
        """Save feature importance or coefficient output for the best model where available."""
        fitted_preprocessor = clf.named_steps["preprocessor"]
        feature_names = self._get_feature_names(fitted_preprocessor)
        fitted_model = clf.named_steps["model"]
        
        if hasattr(fitted_model, "feature_importances_"):
            importance_values = fitted_model.feature_importances_
            importance_df = pd.DataFrame(
                {"feature": feature_names, "importance": importance_values}
            ).sort_values("importance", ascending=False)
            score_col = "importance"
            artifact_name = "best_model_feature_importance.csv"
            chart_name = "best_model_feature_importance_top10.png"
            chart_xlabel = "Importance"
        elif hasattr(fitted_model, "coef_"):
            coefficient_values = fitted_model.coef_[0]
            importance_df = pd.DataFrame(
                {"feature": feature_names, "coefficient": coefficient_values}
            )
            importance_df["absolute_coefficient"] = importance_df["coefficient"].abs()
            importance_df = importance_df.sort_values("absolute_coefficient", ascending=False)
            score_col = "absolute_coefficient"
            artifact_name = "best_model_coefficients.csv"
            chart_name = "best_model_coefficients_top10.png"
            chart_xlabel = "Absolute Coefficient"
        else:
            return
        
        importance_df.to_csv(self.output_dir / artifact_name, index=False)
        
        plt.figure(figsize=(9, 6))
        top_features = importance_df.head(10).sort_values(score_col)
        plt.barh(top_features["feature"], top_features[score_col])
        plt.title(f"Top 10 Predictive Features - {best_model_name}")
        plt.xlabel(chart_xlabel)
        plt.tight_layout()
        plt.savefig(self.output_dir / chart_name, dpi=300)
        plt.close()


def main() -> None:
    """Run model training with default config paths."""
    cleaned_path = PROCESSED_DIR / "teen_mental_health_cleaned.csv"
    trainer = ModelTrainer(
        dataset_path=cleaned_path,
        output_dir=MODEL_OUTPUT_DIR,
        target_column=TARGET_COLUMN,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    trainer.train()


if __name__ == "__main__":
    main()
