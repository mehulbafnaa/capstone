#
# Description:
# This script downloads the 'FrenzyMath/Herald_proofs' dataset
# from the Hugging Face Hub to a specified local directory.
#
# Requirements:
# You need to have the 'datasets' library installed.
# You can install it using pip:
#   pip install datasets
#

from datasets import load_dataset
import pandas as pd
import os

def download_and_inspect_dataset(dataset_name: str, download_path: str, split: str = "train", num_samples: int = 5):
    """
    Downloads a specified dataset from Hugging Face to a local directory.

    Args:
        dataset_name (str): The name of the dataset on Hugging Face Hub.
        download_path (str): The local path to download the dataset to.
        split (str): The dataset split to download (e.g., 'train').
        num_samples (int): The number of samples to display.
    """
    print(f"Attempting to download '{dataset_name}' to '{os.path.abspath(download_path)}'...")

    try:
        # Load the dataset, specifying the cache directory
        # This forces the download to the specified path.
        dataset = load_dataset(dataset_name, split=split, cache_dir=download_path)
        print("\n✅ Dataset downloaded successfully!")

        # Show dataset information
        print("\n--- Dataset Info ---")
        print(dataset)

        # Show a few examples
        print(f"\n--- {num_samples} Example Samples ---")
        
        # Take a small sample to inspect
        samples = dataset.select(range(num_samples))
        
        # Convert to a pandas DataFrame for nice printing
        df = pd.DataFrame(samples)
        
        # To prevent long text from being truncated in the display
        pd.set_option('display.max_colwidth', None)
        print(df.to_string())

    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        print("Please check the following:")
        print("1. You have a stable internet connection.")
        print(f"2. The dataset name '{dataset_name}' is correct and public.")
        print("3. The 'datasets' library is installed correctly.")


if __name__ == "__main__":
    # Specify the correct dataset name for the Herald proofs
    HERALD_DATASET = "FrenzyMath/Herald_proofs"
    
    # Specify the download directory. This will create a folder named
    # 'herald_proofs_data' in the same directory as the script.
    DOWNLOAD_DIRECTORY = "./herald_proofs_data"

    # Run the download and inspection function
    download_and_inspect_dataset(HERALD_DATASET, download_path=DOWNLOAD_DIRECTORY)
