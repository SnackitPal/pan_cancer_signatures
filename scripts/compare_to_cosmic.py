import pandas as pd
import numpy as np
import argparse
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt
import os

def main():
    parser = argparse.ArgumentParser(description="Compare discovered mutational signatures to COSMIC signatures.")
    parser.add_argument("--discovered_profiles_path", type=str, required=True,
                        help="Path to the discovered signature profiles CSV file (e.g., from train_lda_model.py).")
    parser.add_argument("--cosmic_profiles_path", type=str, required=True,
                        help="Path to the COSMIC signature profiles TSV/TXT file (e.g., COSMIC_v3.4_SBS_GRCh38.txt).")
    parser.add_argument("--output_csv_path", type=str, required=True,
                        help="Path for the output CSV file storing cosine similarities.")
    parser.add_argument("--output_heatmap_path", type=str, default=None,
                        help="Optional path for the output heatmap image (e.g., heatmap.png). If not provided, heatmap is not saved.")
    parser.add_argument("--top_n_matches", type=int, default=3,
                        help="Number of top N COSMIC matches to report for each discovered signature (default: 3).")
    
    args = parser.parse_args()

    # Print input parameters
    print("Starting signature comparison script with the following parameters:")
    print(f"  Discovered profiles path: {args.discovered_profiles_path}")
    print(f"  COSMIC profiles path: {args.cosmic_profiles_path}")
    print(f"  Output CSV path: {args.output_csv_path}")
    print(f"  Output heatmap path: {args.output_heatmap_path if args.output_heatmap_path else 'Not saving heatmap'}")
    print(f"  Top N matches: {args.top_n_matches}")

    # a. Load Discovered Signatures
    print(f"\nLoading discovered signature profiles from: {args.discovered_profiles_path}")
    try:
        df_discovered = pd.read_csv(args.discovered_profiles_path, index_col=0)
    except FileNotFoundError:
        print(f"Error: Discovered profiles file not found at {args.discovered_profiles_path}")
        return
    except Exception as e:
        print(f"Error loading discovered profiles CSV file {args.discovered_profiles_path}: {e}")
        return

    if df_discovered.empty:
        print(f"Error: Discovered profiles file at {args.discovered_profiles_path} is empty.")
        return
    print(f"Loaded discovered profiles with shape: {df_discovered.shape}")
    # Expected: rows=signatures, cols=96 contexts

    # b. Load COSMIC Signatures
    print(f"\nLoading COSMIC signature profiles from: {args.cosmic_profiles_path}")
    try:
        # COSMIC files are often TSV, and the first column (MutationType or context) is the index
        df_cosmic = pd.read_csv(args.cosmic_profiles_path, sep='\t', index_col=0)
    except FileNotFoundError:
        print(f"Error: COSMIC profiles file not found at {args.cosmic_profiles_path}")
        return
    except Exception as e:
        print(f"Error loading COSMIC profiles file {args.cosmic_profiles_path}: {e}")
        return

    if df_cosmic.empty:
        print(f"Error: COSMIC profiles file at {args.cosmic_profiles_path} is empty.")
        return

    # Transpose COSMIC DataFrame: COSMIC usually has signatures as columns and contexts as rows.
    # We want signatures as rows and contexts as columns, similar to discovered_profiles.
    df_cosmic = df_cosmic.T
    print(f"Loaded and transposed COSMIC profiles with shape: {df_cosmic.shape}")
    # Expected: rows=COSMIC signatures, cols=contexts (may not be 96, or may be named differently)

    # Normalize COSMIC Profiles (if they don't already sum to 1 per signature)
    # It's common for COSMIC profiles to be provided as probabilities summing to 1.
    # A small tolerance is used for floating point comparisons.
    if not np.allclose(df_cosmic.sum(axis=1), 1.0, atol=1e-3): # Increased tolerance slightly
        print("Normalizing COSMIC signature profiles (rows to sum to 1)...")
        df_cosmic = df_cosmic.apply(lambda x: x / x.sum(), axis=1)
        # Verify normalization
        if not np.allclose(df_cosmic.sum(axis=1), 1.0, atol=1e-3):
            print("Warning: Normalization of COSMIC profiles failed to make rows sum to 1.")
        else:
            print("COSMIC profiles normalized successfully.")
    else:
        print("COSMIC profiles appear to be already normalized (rows sum to ~1).")


    # c. Context Alignment
    print("\nAligning contexts between discovered and COSMIC profiles...")
    discovered_contexts = df_discovered.columns.tolist()
    cosmic_contexts = df_cosmic.columns.tolist()

    common_contexts = sorted(list(set(discovered_contexts) & set(cosmic_contexts)))

    # Validation of common contexts
    if not common_contexts: # Handles len == 0
        print("Error: No common contexts found between discovered and COSMIC profiles. Cannot proceed with comparison.")
        return
    
    expected_contexts_count = df_discovered.shape[1] # Typically 96 for discovered profiles
    if len(common_contexts) < expected_contexts_count:
        print(f"Warning: Number of common contexts ({len(common_contexts)}) is less than the number of contexts in discovered profiles ({expected_contexts_count}). "
              "Signatures will be compared using these common contexts only.")
    
    if len(common_contexts) < 10: # Arbitrary threshold for "too few"
        print(f"Error: Only {len(common_contexts)} common contexts found. This might be too few for a meaningful comparison. Please check your input files.")
        return

    # Reindex DataFrames to common contexts
    df_discovered_aligned = df_discovered[common_contexts]
    df_cosmic_aligned = df_cosmic[common_contexts]

    print(f"Contexts aligned. Using {len(common_contexts)} common contexts for comparison.")

    # d. Calculate Cosine Similarity
    print("\nCalculating cosine similarity between discovered and COSMIC signatures...")
    # Extract numpy arrays for calculation
    # Ensure no NaN values, which can occur if reindexing introduced NaNs (though 'common_contexts' should prevent this)
    # However, original data might have NaNs if not perfectly clean.
    # Cosine similarity function in sklearn handles NaNs by typically erroring or producing NaNs.
    # It's good practice to ensure data is clean before this step.
    # For now, assuming data is clean as per earlier normalization/loading.
    discovered_array = df_discovered_aligned.values
    cosmic_array = df_cosmic_aligned.values
    
    try:
        similarity_matrix = cosine_similarity(discovered_array, cosmic_array)
    except Exception as e:
        print(f"Error calculating cosine similarity: {e}")
        return

    # e. Store and Output Similarity Matrix
    df_similarity = pd.DataFrame(
        similarity_matrix,
        index=df_discovered_aligned.index, # Rows are discovered signatures
        columns=df_cosmic_aligned.index    # Columns are COSMIC signatures
    )

    print(f"\nSaving cosine similarity matrix to: {args.output_csv_path}")
    try:
        # Ensure output directory for the CSV exists (if it's in a subdir)
        output_csv_dir = os.path.dirname(args.output_csv_path)
        if output_csv_dir and not os.path.exists(output_csv_dir):
            os.makedirs(output_csv_dir, exist_ok=True)
            print(f"Created directory for output CSV: {output_csv_dir}")
            
        df_similarity.to_csv(args.output_csv_path)
        print(f"Successfully saved similarity matrix to {args.output_csv_path}")
    except IOError as e:
        print(f"Error: Could not save similarity matrix to {args.output_csv_path}. Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while saving the similarity matrix: {e}")

    # f. Report Top N Matches
    print(f"\nTop {args.top_n_matches} COSMIC Matches for each Discovered Signature:")
    for discovered_sig_name, similarities_series in df_similarity.iterrows():
        top_matches = similarities_series.sort_values(ascending=False)
        
        match_strings = []
        for cosmic_sig, score in top_matches.head(args.top_n_matches).items():
            match_strings.append(f"{cosmic_sig} (Similarity: {score:.3f})")
        
        print(f"  Discovered {discovered_sig_name}: {', '.join(match_strings)}")

    # g. Generate Heatmap (Optional)
    if args.output_heatmap_path:
        print(f"\nGenerating heatmap and saving to: {args.output_heatmap_path}")
        try:
            # Ensure output directory for the heatmap exists
            output_heatmap_dir = os.path.dirname(args.output_heatmap_path)
            if output_heatmap_dir and not os.path.exists(output_heatmap_dir):
                os.makedirs(output_heatmap_dir, exist_ok=True)
                print(f"Created directory for output heatmap: {output_heatmap_dir}")

            # Determine figure size dynamically
            # Adjust these multipliers as needed for optimal visualization
            fig_width = max(10, df_similarity.shape[1] * 0.7) 
            fig_height = max(8, df_similarity.shape[0] * 0.5)
            
            # Limit width and height to avoid excessively large figures
            MAX_FIG_WIDTH = 40 
            MAX_FIG_HEIGHT = 30
            fig_width = min(fig_width, MAX_FIG_WIDTH)
            fig_height = min(fig_height, MAX_FIG_HEIGHT)

            plt.figure(figsize=(fig_width, fig_height))
            
            # Determine annotation size based on number of columns (COSMIC sigs)
            annot_size = 8
            if df_similarity.shape[1] > 30: # If many COSMIC sigs, reduce annot size
                 annot_size = 6
            if df_similarity.shape[1] > 50:
                 annot_size = 4 # even smaller
            
            annot_display = True
            if df_similarity.shape[1] > 70 : # If too many, turn off annotations
                annot_display = False


            sns.heatmap(df_similarity, annot=annot_display, fmt=".2f", cmap="viridis", 
                        linewidths=.5, cbar_kws={'label': 'Cosine Similarity'},
                        annot_kws={"size": annot_size})
            
            plt.title(f"Cosine Similarity: Discovered vs. COSMIC Signatures", fontsize=16) # Simplified title
            plt.xlabel("COSMIC Signatures", fontsize=12)
            plt.ylabel("Discovered Signatures", fontsize=12)
            
            # Adjust tick label sizes and rotation
            plt.xticks(rotation=90, ha='right', fontsize=max(6, 10 - df_similarity.shape[1] // 10)) # Dynamic font size
            plt.yticks(rotation=0, fontsize=max(6, 10 - df_similarity.shape[0] // 5))   # Dynamic font size

            plt.tight_layout()
            plt.savefig(args.output_heatmap_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Successfully saved heatmap to {args.output_heatmap_path}")
        except Exception as e:
            print(f"Error generating or saving heatmap: {e}")
    else:
        print("\nHeatmap generation skipped as --output_heatmap_path was not provided.")

    print("\nSignature comparison script finished.")


if __name__ == "__main__":
    main()
