from DataProcessing.DataSetDownloader import DatasetDownloader
from DataProcessing.DataPreprocessor import DataPreprocessor
from DataAnalysis.EDA import EDA
from DataAnalysis.DataModeling import ModelTrainer
import pandas as pd

# Download dataset
downloader = DatasetDownloader()
raw_data_path = downloader.download_dataset("algozee/teenager-menthal-healy")

# Preprocess dataset
preprocessor = DataPreprocessor(raw_data_path)
cleaned_path, summary_path = preprocessor.preprocess()

# Perform EDA
eda = EDA(dataset_path=cleaned_path)
summary = eda.analyze()

# Train models and compare
trainer = ModelTrainer(dataset_path=cleaned_path)
comparison_df, best_model_name = trainer.train()