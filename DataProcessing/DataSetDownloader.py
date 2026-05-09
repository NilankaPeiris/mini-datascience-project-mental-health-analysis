import os
from dotenv import load_dotenv
import requests
import zipfile

load_dotenv()


class DatasetDownloader:
    """Download and extract Kaggle datasets."""
    
    def __init__(self, api_key=None, data_dir=None):
        """
        Initialize the downloader.
        
        Args:
            api_key (str): Kaggle API key. If None, reads from KAGGLE_API_KEY env var.
            data_dir (str): Directory to save datasets. If None, uses ../data relative to script.
        """
        self.api_key = api_key or os.getenv("KAGGLE_API_KEY")
        self.base_url = "https://www.kaggle.com/api/v1/datasets/download"
        
        if data_dir is None:
            data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
        
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
    
    def download_dataset(self, dataset_identifier):
        """
        Download and extract a dataset from Kaggle.
        
        Args:
            dataset_identifier (str): Dataset identifier in format 'username/dataset-name'
            
        Returns:
            str: Path to the extracted CSV file if successful, None otherwise
        """
        url = f"{self.base_url}/{dataset_identifier}"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        print(f"Downloading dataset: {dataset_identifier}...")
        
        try:
            response = requests.get(url, headers=headers)
            
            if response.status_code == 200:
                zip_filename = dataset_identifier.split('/')[-1] + ".zip"
                zip_path = os.path.join(self.data_dir, zip_filename)
                
                with open(zip_path, 'wb') as f:
                    f.write(response.content)
                
                print(f"Extracting to {self.data_dir}...")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(self.data_dir)
                
                # Find CSV file
                csv_path = None
                for root, dirs, files in os.walk(self.data_dir):
                    for file in files:
                        if file.endswith('.csv'):
                            csv_path = os.path.join(root, file)
                            break
                    if csv_path:
                        break
                
                if csv_path:
                    print(f"Dataset downloaded and extracted successfully!")
                    print(f"CSV file found at: {csv_path}")
                    return csv_path
                else:
                    print("Dataset extracted but no CSV file found!")
                    return None
            else:
                print(f"Error downloading dataset: {response.status_code}")
                print(response.text)
                return None
        except Exception as e:
            print(f"Error: {e}")
            return None


# if __name__ == "__main__":
#     downloader = DatasetDownloader()
#     downloader.download_dataset("algozee/teenager-menthal-healy")