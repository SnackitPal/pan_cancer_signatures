import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Analyze patient signature exposures and generate cohort-level plots.")
    parser.add_argument("--exposures_file", type=str, required=True,
                        help="Path to the patient exposures CSV file (e.g., patient_exposures_k*.csv).")
    parser.add_argument("--sample_cohort_map_file", type=str, required=True,
                        help="Path to the sample-to-cohort mapping CSV file (Tumor_Sample_Barcode, Cohort).")
    parser.add_argument("--output_dir_figures", type=str, required=True,
                        help="Path to the directory where generated figures will be saved.")
    
    args = parser.parse_args()

    # Print input parameters
    print("Starting patient exposure analysis script with the following parameters:")
    print(f"  Exposures file path: {args.exposures_file}")
    print(f"  Sample-cohort map file path: {args.sample_cohort_map_file}")
    print(f"  Output directory for figures: {args.output_dir_figures}")

    # a. Load Exposures Data
    print(f"\nLoading patient exposures from: {args.exposures_file}")
    try:
        # Assuming the first column is 'Tumor_Sample_Barcode' or similar sample ID
        df_exposures = pd.read_csv(args.exposures_file, index_col=0)
    except FileNotFoundError:
        print(f"Error: Exposures file not found at {args.exposures_file}")
        return
    except Exception as e:
        print(f"Error loading exposures CSV file {args.exposures_file}: {e}")
        return

    if df_exposures.empty:
        print(f"Error: Exposures file at {args.exposures_file} is empty.")
        return
    print(f"Loaded exposures data with shape: {df_exposures.shape}")
    # Store the original index name if it's not None, otherwise assume 'Tumor_Sample_Barcode'
    exposure_index_name = df_exposures.index.name if df_exposures.index.name else 'Tumor_Sample_Barcode'


    # b. Load Sample-Cohort Map
    print(f"\nLoading sample-cohort map from: {args.sample_cohort_map_file}")
    try:
        df_cohort_map = pd.read_csv(args.sample_cohort_map_file)
    except FileNotFoundError:
        print(f"Error: Sample-cohort map file not found at {args.sample_cohort_map_file}")
        return
    except Exception as e:
        print(f"Error loading sample-cohort map CSV file {args.sample_cohort_map_file}: {e}")
        return

    if df_cohort_map.empty:
        print(f"Error: Sample-cohort map file at {args.sample_cohort_map_file} is empty.")
        return
    print(f"Loaded sample-cohort map with shape: {df_cohort_map.shape}")

    # Ensure expected columns are in cohort map
    if 'Tumor_Sample_Barcode' not in df_cohort_map.columns or 'Cohort' not in df_cohort_map.columns:
        print(f"Error: Sample-cohort map file {args.sample_cohort_map_file} must contain 'Tumor_Sample_Barcode' and 'Cohort' columns.")
        return

    # c. Merge DataFrames
    print("\nMerging exposures data with cohort map...")
    
    # Reset index of df_exposures to make 'Tumor_Sample_Barcode' (or its actual name) a column for merging
    df_exposures_reset = df_exposures.reset_index()
    # Rename the index column to 'Tumor_Sample_Barcode' if it was different, to ensure consistent merge key
    if exposure_index_name != 'Tumor_Sample_Barcode' and exposure_index_name in df_exposures_reset.columns:
        df_exposures_reset.rename(columns={exposure_index_name: 'Tumor_Sample_Barcode'}, inplace=True)
    
    rows_before_merge = len(df_exposures_reset)
    merged_df = pd.merge(df_exposures_reset, df_cohort_map, on='Tumor_Sample_Barcode', how='left')
    rows_after_merge = len(merged_df)

    print(f"Rows before merge (exposures): {rows_before_merge}, Rows after merge: {rows_after_merge}")

    # Handle Merge Issues
    missing_in_map = merged_df[merged_df['Cohort'].isnull()]
    if not missing_in_map.empty:
        print(f"Warning: {len(missing_in_map)} samples from exposures file were not found in the cohort map.")
        if len(missing_in_map) < 10: # Print a few examples if the list is short
            print("  Examples of samples with missing cohort information:")
            for sample_id in missing_in_map['Tumor_Sample_Barcode'].head().tolist():
                print(f"    - {sample_id}")
        # Drop rows with missing cohort information
        merged_df.dropna(subset=['Cohort'], inplace=True)
        print(f"Dropped {len(missing_in_map)} samples with missing cohort information.")
    
    if merged_df.empty:
        print("Error: No samples remaining after merging and dropping those with missing cohort information. Cannot proceed.")
        return

    print(f"Shape of final merged DataFrame: {merged_df.shape}")

    # d. Create Output Directory
    print(f"\nEnsuring output directory for figures exists: {args.output_dir_figures}")
    try:
        os.makedirs(args.output_dir_figures, exist_ok=True)
        print(f"Output directory ready: {args.output_dir_figures}")
    except OSError as e:
        print(f"Error creating output directory {args.output_dir_figures}: {e}")
        return # Exit if directory cannot be created

    # Identify Signature Columns
    # These are columns not 'Tumor_Sample_Barcode' or 'Cohort'
    non_signature_cols = ['Tumor_Sample_Barcode', 'Cohort']
    signature_columns_list = [col for col in merged_df.columns if col not in non_signature_cols]

    if not signature_columns_list:
        print("Error: No signature columns found in the merged data. Cannot proceed with analysis.")
        return

    print(f"\nIdentified {len(signature_columns_list)} signature columns for analysis: {', '.join(signature_columns_list[:5])}...")

    # Generate Boxplots per Signature
    print("\nGenerating boxplots of signature exposures per cohort...")
    for signature_column_name in signature_columns_list:
        plt.figure(figsize=(10, 7))
        try:
            sns.boxplot(x='Cohort', y=signature_column_name, data=merged_df, palette='viridis')
            plt.title(f"Exposure of {signature_column_name} across Cohorts", fontsize=16)
            plt.ylabel("Signature Exposure", fontsize=12)
            plt.xlabel("Cancer Cohort", fontsize=12)
            
            # Rotate x-axis labels if cohort names are many or long
            num_cohorts = merged_df['Cohort'].nunique()
            if num_cohorts > 10 or merged_df['Cohort'].str.len().max() > 10:
                plt.xticks(rotation=45, ha='right')
            else:
                plt.xticks(rotation=0) # Keep horizontal if few and short

            plt.tight_layout()
            
            plot_filename = os.path.join(args.output_dir_figures, f"exposure_boxplot_{signature_column_name}.png")
            plt.savefig(plot_filename, dpi=300)
            plt.close() # Close the figure to free memory
            print(f"  Saved boxplot to {plot_filename}")

        except Exception as e:
            print(f"  Error generating or saving boxplot for {signature_column_name}: {e}")
            if 'plt' in locals() and plt.gcf().get_axes(): # Check if a figure is open
                plt.close() # Attempt to close if an error occurred mid-plot

    # Calculate and Print Summary Statistics
    print("\nSummary Statistics of Signature Exposures per Cohort:")
    try:
        # Group by 'Cohort' and calculate mean and median for all signature columns
        summary_stats = merged_df.groupby('Cohort')[signature_columns_list].agg(['mean', 'median'])
        
        # Print the summary_stats DataFrame
        # For better console output, especially if wide, consider specific formatting
        # For now, direct print, or convert to string with options
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
            print(summary_stats)
            
    except Exception as e:
        print(f"Error calculating or printing summary statistics: {e}")

    # Generate Stacked Bar Plot of Average Signature Contributions
    print("\nGenerating stacked bar plot of average signature contributions per cohort...")
    try:
        # 1. Calculate Mean Exposures per Cohort
        if not signature_columns_list: # Should have been caught earlier, but as a safeguard
            print("Error: Cannot generate stacked bar plot as no signature columns were identified.")
            return

        mean_exposures_by_cohort = merged_df.groupby('Cohort')[signature_columns_list].mean()

        if mean_exposures_by_cohort.empty:
            print("Warning: Mean exposures by cohort are empty. Skipping stacked bar plot generation.")
        else:
            # 2. Generate Stacked Bar Plot
            plt.figure() # Create a new figure instance before plotting
            ax = mean_exposures_by_cohort.plot(
                kind='bar', 
                stacked=True, 
                figsize=(14, 9), # Slightly adjusted for potentially many cohorts/signatures
                colormap='viridis' # Using a perceptually uniform colormap
            )
            
            plt.title("Average Signature Contributions per Cancer Cohort", fontsize=16)
            plt.ylabel("Mean Signature Exposure", fontsize=12)
            plt.xlabel("Cancer Cohort", fontsize=12)
            
            # Rotate x-axis labels (always a good idea if cohort names can vary)
            plt.xticks(rotation=45, ha='right') 
            
            # Legend placement
            # Adjust legend to be outside if many signatures, otherwise default placement
            num_signatures = len(signature_columns_list)
            if num_signatures > 10: # Arbitrary threshold to move legend outside
                plt.legend(title='Signatures', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='small')
                plt.tight_layout(rect=[0, 0, 0.85, 1]) # Make space for external legend
            else:
                plt.legend(title='Signatures', fontsize='small')
                plt.tight_layout()

            stacked_plot_filename = os.path.join(args.output_dir_figures, "avg_signature_contributions_by_cohort.png")
            plt.savefig(stacked_plot_filename, dpi=300, bbox_inches='tight') # Use bbox_inches for tight layout saving
            plt.close() # Close the figure
            print(f"  Saved stacked bar plot to {stacked_plot_filename}")

    except Exception as e:
        print(f"  Error generating or saving stacked bar plot of average contributions: {e}")
        if 'plt' in locals() and plt.gcf().get_axes(): # Check if a figure is open
            plt.close() # Attempt to close if an error occurred mid-plot


    print("\nPatient exposure analysis finished.")

if __name__ == "__main__":
    main()
