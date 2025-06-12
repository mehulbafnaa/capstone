# #
# # Description:
# # This script loads the 'FrenzyMath/Herald_proofs' dataset and prints its
# # schema (features). This is the first step to understand the available
# # columns before performing any detailed analysis.
# #
# # Requirements:
# # You need to have the 'datasets' library installed.
# #   pip install datasets
# #

# from datasets import load_dataset

# def discover_dataset_schema(dataset_name: str, split: str = "train"):
#     """
#     Loads a dataset and prints its features to reveal the column names and types.

#     Args:
#         dataset_name (str): The name of the dataset on Hugging Face Hub.
#         split (str): The dataset split to load (e.g., 'train').
#     """
#     print(f"--- Discovering schema for '{dataset_name}' ---")

#     try:
#         # 1. Load the dataset
#         print("\n[1/2] Loading dataset metadata...")
#         dataset = load_dataset(dataset_name, split=split, trust_remote_code=True)
#         print("✅ Dataset loaded successfully.")

#         # 2. Print the features (schema)
#         print("\n[2/2] Dataset Schema (Features):")
#         print(dataset.features)
#         print("\n--- End of Schema Discovery ---")
#         print("\nNow that we have the column names, we can proceed with a detailed analysis.")


#     except Exception as e:
#         print(f"\n❌ An error occurred: {e}")
#         print("Please check the following:")
#         print("1. You have a stable internet connection.")
#         print(f"2. The dataset name '{dataset_name}' is correct and public.")
#         print("3. The 'datasets' library is installed correctly.")


# if __name__ == "__main__":
#     # Specify the correct dataset name for the Herald proofs
#     HERALD_DATASET = "FrenzyMath/Herald_proofs"
    
#     # Run the schema discovery function
#     discover_dataset_schema(HERALD_DATASET)





#
# Description:
# This script performs an in-depth analysis of the 'FrenzyMath/Herald_proofs'
# dataset, using the now-known schema. The goal is to understand the structure,
# content, and relationships between all relevant columns to inform the design
# of a proof generation inference pipeline.
#
# Requirements:
# You need to have the 'datasets', 'pandas', and 'matplotlib' libraries installed.
#   pip install datasets pandas matplotlib seaborn
#

from datasets import load_dataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def investigate_herald_dataset(dataset_name: str, split: str = "train"):
    """
    Downloads, analyzes, and displays key insights from the Herald dataset.

    Args:
        dataset_name (str): The name of the dataset on Hugging Face Hub.
        split (str): The dataset split to analyze (e.g., 'train').
    """
    print(f"--- Starting Investigation of '{dataset_name}' ---")

    try:
        # 1. Load the dataset
        print("\n[1/4] Loading dataset...")
        dataset = load_dataset(dataset_name, split=split, trust_remote_code=True)
        print("✅ Dataset downloaded successfully.")

        # Convert to pandas DataFrame for easier analysis
        df = dataset.to_pandas()

        # 2. Confirm schema and basic info
        print("\n[2/4] Confirming schema and basic info...")
        print("Dataset Features (Columns):")
        print(dataset.features)
        print(f"\nTotal number of examples: {len(df)}")
        
        # 3. Analyze content and length statistics for all relevant columns
        print("\n[3/4] Analyzing content and length statistics...")
        df['formal_theorem_len'] = df['formal_theorem'].str.len()
        df['formal_proof_len'] = df['formal_proof'].str.len()
        df['informal_theorem_len'] = df['informal_theorem'].str.len()
        df['header_len'] = df['header'].str.len()


        print("\n--- Length Statistics ---")
        print(df[['formal_theorem_len', 'formal_proof_len', 'informal_theorem_len', 'header_len']].describe().round(2))

        # Analyze common proof starting words (tactics)
        df['proof_start_word'] = df['formal_proof'].str.split().str.get(0)
        common_tactics = df['proof_start_word'].value_counts().nlargest(10)
        print("\n--- Top 10 Most Common Proof Starting Words (Tactics) ---")
        print(common_tactics)

        # Visualize the length distributions
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        sns.histplot(df['formal_theorem_len'], ax=axes[0, 0], kde=True, color='skyblue')
        axes[0, 0].set_title('Distribution of Formal Theorem Lengths')
        
        sns.histplot(df['formal_proof_len'], ax=axes[0, 1], kde=True, color='salmon')
        axes[0, 1].set_title('Distribution of Formal Proof Lengths')

        sns.histplot(df['informal_theorem_len'], ax=axes[1, 0], kde=True, color='lightgreen')
        axes[1, 0].set_title('Distribution of Informal Theorem Lengths')

        sns.histplot(df['header_len'], ax=axes[1, 1], kde=True, color='gold')
        axes[1, 1].set_title('Distribution of Header Lengths')

        plt.suptitle('Dataset Content Analysis', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save the plot to a file
        plot_filename = "dataset_length_distribution.png"
        plt.savefig(plot_filename)
        print(f"\nLength distribution plot saved to '{plot_filename}'")
        plt.show()

        # 4. Display diverse, full examples
        print("\n[4/4] Displaying complete examples for qualitative review...")
        # Select a few samples to display, including one with a longer proof
        sample_indices = [5, 15, df['formal_proof_len'].idxmax()]

        for i, index in enumerate(sample_indices):
            example = df.loc[index]
            print("\n" + "="*80)
            print(f"--- EXAMPLE {i+1} (Dataset index: {index}) ---")
            print("="*80)

            print("\n>>> HEADER (Context for Inference):")
            print("-" * 40)
            print(example['header'])

            print("\n>>> INFORMAL THEOREM (Context for Inference):")
            print("-" * 40)
            print(example['informal_theorem'])

            print("\n>>> FORMAL THEOREM (Input for Inference):")
            print("-" * 40)
            print(example['formal_theorem'])
            
            print("\n>>> FORMAL PROOF (Target for Generation):")
            print("-" * 40)
            print(example['formal_proof'])
            print("="*80)

    except Exception as e:
        print(f"\n❌ An error occurred: {e}")


if __name__ == "__main__":
    # Specify the correct dataset name for the Herald proofs
    HERALD_DATASET = "FrenzyMath/Herald_proofs"
    
    # Run the investigation function
    investigate_herald_dataset(HERALD_DATASET)

