import pandas as pd
import numpy as np
import argparse
from sklearn.decomposition import LatentDirichletAllocation
import joblib
import os

def main():
    parser = argparse.ArgumentParser(description="Train an LDA model on a mutation catalog to discover mutational signatures.")
    parser.add_argument("--matrix_path", type=str, required=True,
                        help="Path to the input mutation catalog CSV file (samples x 96 contexts).")
    parser.add_argument("--k", type=int, required=True,
                        help="Number of mutational signatures (topics) to discover.")
    parser.add_argument("--output_dir_models", type=str, required=True,
                        help="Path to the directory where output models and results will be saved.")
    parser.add_argument("--random_seed", type=int, default=42,
                        help="Random seed for LDA reproducibility (default: 42).")
    
    args = parser.parse_args()

    # Print input parameters
    print("Starting LDA model training script with the following parameters:")
    print(f"  Input matrix path: {args.matrix_path}")
    print(f"  Number of signatures (k): {args.k}")
    print(f"  Output directory for models: {args.output_dir_models}")
    print(f"  Random seed: {args.random_seed}")

    # Data Loading and Preparation
    print(f"\nLoading mutation catalog from: {args.matrix_path}")
    try:
        mutation_df = pd.read_csv(args.matrix_path)
    except FileNotFoundError:
        print(f"Error: Input matrix file not found at {args.matrix_path}")
        return # Exit script
    except Exception as e:
        print(f"Error loading CSV file {args.matrix_path}: {e}")
        return

    if mutation_df.empty:
        print(f"Error: Input matrix at {args.matrix_path} is empty.")
        return

    # Set the first column as index (expected to be 'Tumor_Sample_Barcode' or similar)
    try:
        sample_id_col_name = mutation_df.columns[0]
        mutation_df.set_index(sample_id_col_name, inplace=True)
        print(f"Set '{sample_id_col_name}' as index.")
    except IndexError:
        print("Error: Could not identify the first column to set as index. The CSV file might be empty or malformed.")
        return
        
    # The remaining columns constitute the feature matrix (96 mutation contexts)
    # Assuming the first column was the sample ID, all other columns are features.
    mutation_matrix = mutation_df.copy() # .copy() if further modifications to mutation_df are not intended to affect mutation_matrix

    # Input Validation
    # Check if the matrix contains any negative values
    if (mutation_matrix < 0).any().any():
        print("Error: Mutation matrix contains negative values. LDA expects non-negative counts.")
        return # Exit script
    
    # Check if all values are integers (or float that are whole numbers)
    # LDA works with counts, so non-integer values might indicate issues or require conversion.
    # For now, we'll just check for negativity. A more robust check for integer counts might be needed.
    # Example: if not mutation_matrix.applymap(lambda x: x == int(x) if pd.notnull(x) else True).all().all():
    #    print("Warning: Mutation matrix contains non-integer values. LDA works best with integer counts.")

    print(f"Loaded mutation matrix with shape: {mutation_matrix.shape}")
    if mutation_matrix.shape[1] != 96:
        print(f"Warning: The loaded matrix has {mutation_matrix.shape[1]} columns. Expecting 96 columns for standard trinucleotide contexts.")
        # Depending on strictness, one might choose to exit here.

    # LDA Model Training
    print("\nStarting LDA model training...")
    # Initialize LatentDirichletAllocation model
    # Other parameters (e.g., doc_topic_prior, topic_word_prior) use scikit-learn defaults.
    lda_model = LatentDirichletAllocation(
        n_components=args.k,
        random_state=args.random_seed,
        learning_method='batch', # default, but explicit
        max_iter=10              # default, can be increased for production
    )

    # Fit the LDA model to the mutation count matrix
    # sklearn LDA expects samples as rows and features (96 contexts) as columns.
    # Ensure mutation_matrix is suitable (e.g. a numpy array or pandas DataFrame of counts)
    try:
        lda_model.fit(mutation_matrix)
        print("LDA model training completed.")
    except Exception as e:
        print(f"Error during LDA model training: {e}")
        return

    # Output Directory Creation
    try:
        os.makedirs(args.output_dir_models, exist_ok=True)
        print(f"\nCreated output directory: {args.output_dir_models} (or it already existed).")
    except OSError as e:
        print(f"Error creating output directory {args.output_dir_models}: {e}")
        return

    # Save Trained LDA Model
    model_filename = f"lda_model_k{args.k}_seed{args.random_seed}.joblib"
    full_model_path = os.path.join(args.output_dir_models, model_filename)
    try:
        joblib.dump(lda_model, full_model_path)
        print(f"Trained LDA model saved to: {full_model_path}")
    except Exception as e:
        print(f"Error saving LDA model to {full_model_path}: {e}")

    # Extract, Process, and Save Signature Profiles (W matrix / lda_model.components_)
    print("\nExtracting and saving signature profiles...")
    profiles = lda_model.components_
    # Normalize each signature profile (row) so its components sum to 1
    profiles_normalized = profiles / profiles.sum(axis=1, keepdims=True)
    
    df_profiles = pd.DataFrame(
        profiles_normalized,
        index=[f"Signature_{i+1}" for i in range(args.k)],
        columns=mutation_matrix.columns  # Use original 96 context column names
    )
    profiles_filename = f"signature_profiles_k{args.k}_seed{args.random_seed}.csv"
    full_profiles_path = os.path.join(args.output_dir_models, profiles_filename)
    try:
        df_profiles.to_csv(full_profiles_path)
        print(f"Signature profiles saved to: {full_profiles_path}")
    except Exception as e:
        print(f"Error saving signature profiles to {full_profiles_path}: {e}")

    # Extract, Process, and Save Patient Exposures (H matrix / lda_model.transform())
    print("\nExtracting and saving patient exposures...")
    try:
        exposures = lda_model.transform(mutation_matrix)
    except Exception as e:
        print(f"Error transforming data to get patient exposures: {e}")
        return # Cannot proceed if this fails
        
    df_exposures = pd.DataFrame(
        exposures,
        index=mutation_matrix.index,  # Use original sample IDs
        columns=[f"Signature_{i+1}" for i in range(args.k)]
    )
    exposures_filename = f"patient_exposures_k{args.k}_seed{args.random_seed}.csv"
    full_exposures_path = os.path.join(args.output_dir_models, exposures_filename)
    try:
        df_exposures.to_csv(full_exposures_path)
        print(f"Patient exposures saved to: {full_exposures_path}")
    except Exception as e:
        print(f"Error saving patient exposures to {full_exposures_path}: {e}")

    print("\nLDA model training and output saving finished.")

if __name__ == "__main__":
    main()
