import pandas as pd
import numpy as np
import argparse
import os
import glob
from scipy.stats import fisher_exact
import sys

# Custom exception for cleaner error handling, especially for tests
class ScriptLogicError(Exception):
    pass

def create_parser():
    parser = argparse.ArgumentParser(description="Generate GSEA ranked list.")
    parser.add_argument("--exposures_file", type=str, required=True, help="Path to the exposures file.")
    parser.add_argument("--sample_map_file", type=str, required=True, help="Path to the sample map file.")
    parser.add_argument("--maf_input_dir", type=str, required=True, help="Path to the MAF input directory.")
    parser.add_argument("--target_cohort", type=str, required=True, help="Target cohort.")
    parser.add_argument("--target_signature_column", type=str, required=True, help="Target signature column.")
    parser.add_argument("--high_exposure_quantile", type=float, required=True, help="High exposure quantile.")
    parser.add_argument("--low_exposure_quantile", type=float, required=True, help="Low exposure quantile.")
    parser.add_argument("--min_group_size", type=int, required=True, help="Minimum group size.")
    parser.add_argument("--output_ranked_gene_file", type=str, required=True, help="Path to the output ranked gene file.")
    return parser

def load_and_filter_data(args):
    """
    Loads exposures and sample map data, merges them, and filters for the target cohort.
    Returns:
        pd.DataFrame: Filtered cohort DataFrame.
    Raises:
        FileNotFoundError: If data files are not found.
        ValueError: If required columns are missing for merging.
        ScriptLogicError: If no samples are found for the target cohort.
    """
    print(f"Loading exposures file: {args.exposures_file}")
    try:
        exposures_df = pd.read_csv(args.exposures_file, index_col=0) # First column is index
        # Ensure index is named 'Tumor_Sample_Barcode'
        if exposures_df.index.name is None or exposures_df.index.name.lower().replace(" ", "_") != 'tumor_sample_barcode':
             print(f"Info: Exposures DataFrame index name was '{exposures_df.index.name}', renaming to 'Tumor_Sample_Barcode'.")
             exposures_df.index.name = 'Tumor_Sample_Barcode'
    except FileNotFoundError:
        print(f"Error: Exposures file not found at {args.exposures_file}")
        raise 
    except Exception as e:
        print(f"Error loading exposures CSV file {args.exposures_file}: {e}")
        raise
    
    print(f"Loading sample map file: {args.sample_map_file}")
    try:
        sample_map_df = pd.read_csv(args.sample_map_file) # Standard CSV
    except FileNotFoundError:
        print(f"Error: Sample map file not found at {args.sample_map_file}")
        raise
    except Exception as e:
        print(f"Error loading sample map CSV file {args.sample_map_file}: {e}")
        raise

    print("Merging exposures and sample map data...")
    
    # Check for 'Tumor_Sample_Barcode' in sample_map_df
    if 'Tumor_Sample_Barcode' not in sample_map_df.columns:
        msg = (f"Error: Required column 'Tumor_Sample_Barcode' not found in sample map file: {args.sample_map_file}. "
               f"Available columns: {sample_map_df.columns.tolist()}")
        print(msg)
        raise ValueError(msg)

    # Reset index of exposures_df to use 'Tumor_Sample_Barcode' as a column for merging
    exposures_df_to_merge = exposures_df.reset_index()

    # Check if 'Tumor_Sample_Barcode' column now exists in exposures_df_to_merge
    if 'Tumor_Sample_Barcode' not in exposures_df_to_merge.columns:
        msg = (f"Error: 'Tumor_Sample_Barcode' column not found in exposures data after resetting index. "
               f"Original index name was '{exposures_df.index.name}'. Available columns: {exposures_df_to_merge.columns.tolist()}")
        print(msg)
        raise ValueError(msg)

    # Perform the merge using 'Tumor_Sample_Barcode' which should be present in both DataFrames
    merged_df = pd.merge(exposures_df_to_merge, sample_map_df, on='Tumor_Sample_Barcode', how='inner')
    print(f"Data merged successfully. Merged DataFrame shape: {merged_df.shape}")

    print(f"Filtering for target cohort: {args.target_cohort}")
    cohort_df = merged_df[merged_df['Cohort'] == args.target_cohort]
    
    if cohort_df.empty:
        msg = f"Error: No samples found for the target cohort '{args.target_cohort}'. Please check your cohort name and sample map."
        print(msg)
        raise ScriptLogicError(msg)
        
    print(f"Found {len(cohort_df)} samples for cohort '{args.target_cohort}'.")
    return cohort_df

def define_exposure_groups(cohort_df, args):
    """
    Defines high and low exposure groups based on the target signature column and quantiles.
    Returns:
        tuple(pd.DataFrame, pd.DataFrame): (high_exposure_group, low_exposure_group)
    Raises:
        ValueError: If target signature column is not found or quantiles are invalid.
        ScriptLogicError: If group sizes are insufficient.
    """
    print(f"Defining exposure groups based on signature '{args.target_signature_column}'")
    if args.target_signature_column not in cohort_df.columns:
        msg = f"Error: Target signature column '{args.target_signature_column}' not found in the data. Available columns: {cohort_df.columns.tolist()}"
        print(msg)
        raise ValueError(msg)

    if not (0 <= args.low_exposure_quantile < args.high_exposure_quantile <= 1):
        msg = f"Error: Invalid quantile values. Low: {args.low_exposure_quantile}, High: {args.high_exposure_quantile}. Ensure 0 <= low < high <= 1."
        print(msg)
        raise ValueError(msg)

    high_threshold = cohort_df[args.target_signature_column].quantile(args.high_exposure_quantile)
    low_threshold = cohort_df[args.target_signature_column].quantile(args.low_exposure_quantile)

    high_exposure_group = cohort_df[cohort_df[args.target_signature_column] >= high_threshold]
    low_exposure_group = cohort_df[cohort_df[args.target_signature_column] <= low_threshold]

    print(f"High-exposure group (>= {args.high_exposure_quantile*100:.0f}th percentile, threshold >= {high_threshold:.4f}): {len(high_exposure_group)} samples.")
    print(f"Low-exposure group (<= {args.low_exposure_quantile*100:.0f}th percentile, threshold <= {low_threshold:.4f}): {len(low_exposure_group)} samples.")

    if len(high_exposure_group) < args.min_group_size or len(low_exposure_group) < args.min_group_size:
        msg = (f"Warning: One or both exposure groups are smaller than the minimum required size ({args.min_group_size}).\n"
               f"  High-exposure group size: {len(high_exposure_group)}\n"
               f"  Low-exposure group size: {len(low_exposure_group)}\n"
               "Halting analysis as group sizes are insufficient.")
        print(msg)
        raise ScriptLogicError(msg)
    
    print(f"Exposure group sizes are adequate (High: {len(high_exposure_group)}, Low: {len(low_exposure_group)}, Min required: {args.min_group_size}).")
    return high_exposure_group, low_exposure_group

def process_maf_files(args, target_cohort_sample_ids):
    """
    Processes MAF files to build a sample_mutations dictionary and get a set of all genes.
    Returns:
        tuple(dict, set): (sample_mutations, all_genes)
    """
    sample_mutations = {}
    all_genes = set()
    maf_dir_path = os.path.join(args.maf_input_dir, args.target_cohort)
    
    if args.target_cohort == "":
        maf_dir_path = args.maf_input_dir
        
    print(f"Searching for MAF files in: {maf_dir_path}")
    maf_files = glob.glob(os.path.join(maf_dir_path, "*.maf.gz")) + glob.glob(os.path.join(maf_dir_path, "*.maf"))

    if not maf_files:
        print(f"Warning: No MAF files found in {maf_dir_path}. The mutation matrix will be empty.")
    else:
        print(f"Found {len(maf_files)} MAF file(s) to process: {maf_files}")

    for maf_file_path in maf_files:
        print(f"Processing MAF file: {maf_file_path}")
        try:
            maf_df = pd.read_csv(maf_file_path, sep='\t', comment='#', low_memory=False, 
                                 compression='gzip' if maf_file_path.endswith('.gz') else None)
            
            sample_col, gene_col, variant_type_col = 'Tumor_Sample_Barcode', 'Hugo_Symbol', 'Variant_Type'

            if not {sample_col, gene_col, variant_type_col}.issubset(maf_df.columns):
                print(f"  Warning: MAF file {maf_file_path} is missing one or more required columns. Skipping this file.")
                continue

            snv_df = maf_df[maf_df[variant_type_col] == 'SNP']
            if snv_df.empty:
                print(f"  No 'SNP' variants found in {maf_file_path}.")
                continue
            
            print(f"  Found {len(snv_df)} SNP entries in {maf_file_path}.")
            for _, row in snv_df.iterrows():
                sample_id, gene_symbol = row[sample_col], row[gene_col]
                if sample_id in target_cohort_sample_ids:
                    sample_mutations.setdefault(sample_id, set()).add(gene_symbol)
                    all_genes.add(gene_symbol)
        except Exception as e:
            print(f"  Error processing MAF file {maf_file_path}: {e}")
            continue
            
    print(f"Processed all MAF files. Found mutations for {len(sample_mutations)} samples in the target cohort.")
    return sample_mutations, all_genes

def create_binary_mutation_matrix(target_cohort_sample_ids, all_genes, sample_mutations):
    """
    Creates the binary mutation DataFrame.
    Returns:
        pd.DataFrame: The binary mutation matrix.
    """
    print("Creating binary mutation matrix...")
    mutation_matrix_df = pd.DataFrame(0, index=target_cohort_sample_ids, columns=list(all_genes))
    for sample_id, mutated_genes in sample_mutations.items():
        if sample_id in mutation_matrix_df.index:
            for gene in mutated_genes:
                if gene in mutation_matrix_df.columns:
                    mutation_matrix_df.loc[sample_id, gene] = 1
    mutation_matrix_df = mutation_matrix_df.fillna(0).astype(int)
    print(f"Binary mutation matrix created with dimensions: {mutation_matrix_df.shape}")
    return mutation_matrix_df

def perform_differential_analysis(mutation_matrix_df, high_exposure_sample_ids, low_exposure_sample_ids, all_genes):
    """
    Performs differential mutation analysis.
    Returns:
        pd.DataFrame: DataFrame with analysis results.
    """
    if not all_genes:
        print("No genes found in MAF files for the target cohort. Cannot perform differential mutation analysis.")
        return pd.DataFrame(columns=['Gene', 'RankMetric', 'PValue', 'OddsRatio'])

    print(f"\nStarting differential mutation analysis for {len(mutation_matrix_df.columns)} genes...")
    results_list = []
    for gene in mutation_matrix_df.columns:
        n_high_mut = mutation_matrix_df.loc[list(high_exposure_sample_ids & set(mutation_matrix_df.index)), gene].sum()
        n_high_nomut = len(high_exposure_sample_ids) - n_high_mut
        n_low_mut = mutation_matrix_df.loc[list(low_exposure_sample_ids & set(mutation_matrix_df.index)), gene].sum()
        n_low_nomut = len(low_exposure_sample_ids) - n_low_mut

        table = [[n_high_mut, n_high_nomut], [n_low_mut, n_low_nomut]]
        oddsratio, p_value = fisher_exact(table)

        pseudo_table = [[n_high_mut + 0.5, n_high_nomut + 0.5], [n_low_mut + 0.5, n_low_nomut + 0.5]]
        pseudo_oddsratio = (pseudo_table[0][0] * pseudo_table[1][1]) / (pseudo_table[0][1] * pseudo_table[1][0])
        
        safe_p_value = max(p_value, 1e-300)
        rank_metric = -np.log10(safe_p_value) * np.sign(np.log(pseudo_oddsratio)) if pseudo_oddsratio != 0 else 0
        
        results_list.append({'Gene': gene, 'RankMetric': rank_metric, 'PValue': p_value, 'OddsRatio': oddsratio})

    print(f"Differential mutation analysis completed. Tested {len(mutation_matrix_df.columns)} genes.")
    return pd.DataFrame(results_list)

def save_ranked_list(results_df, output_file):
    """Saves the ranked list to a file."""
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir): # Check if output_dir is not empty string
        try:
            os.makedirs(output_dir, exist_ok=True)
            print(f"Created output directory: {output_dir}")
        except Exception as e:
            print(f"Error creating output directory {output_dir}: {e}")
            raise ScriptLogicError(f"Failed to create output directory: {e}")

    if results_df.empty:
        print("No results to output. The ranked gene list will be empty or reflect no found genes.")
        # Create an empty file if specified, or a file with headers if that's preferred
        with open(output_file, 'w') as f:
            # For GSEA .rnk, it's typically two columns, no header.
            # If results_df is empty because no genes, this makes an empty file.
            # If results_df is empty due to no significant results but genes exist,
            # it will still write an empty file if we don't select columns.
             pass # Creates an empty file
        print(f"Empty ranked gene file created at: {output_file}")
        return

    # Handle NaNs in RankMetric if any; GSEA might not like them.
    results_df.dropna(subset=['RankMetric'], inplace=True)
    results_df.sort_values(by='RankMetric', ascending=False, inplace=True)
    
    ranked_list_df = results_df[['Gene', 'RankMetric']]
    try:
        ranked_list_df.to_csv(output_file, sep='\t', index=False, header=False)
        print(f"Ranked gene list saved to: {output_file}")
    except Exception as e:
        print(f"Error saving ranked gene list to {output_file}: {e}")
        # Decide if this should re-raise or sys.exit depending on script vs library use
        # For now, just prints, but in main, it would lead to sys.exit implicitly or explicitly.
        raise ScriptLogicError(f"Failed to save output file: {e}")


def main(args_list=None):
    parser = create_parser()
    args = parser.parse_args(args_list)

    try:
        cohort_df = load_and_filter_data(args)
        high_exposure_group, low_exposure_group = define_exposure_groups(cohort_df, args)
    except FileNotFoundError:
        sys.exit(1) # Error message already printed
    except (ValueError, ScriptLogicError) as e:
        # Error messages already printed by the functions
        sys.exit(1)
        
    target_cohort_sample_ids = list(cohort_df['Tumor_Sample_Barcode'].unique())
    high_exposure_sample_ids = set(high_exposure_group['Tumor_Sample_Barcode'])
    low_exposure_sample_ids = set(low_exposure_group['Tumor_Sample_Barcode'])
    
    print(f"Identified {len(target_cohort_sample_ids)} unique sample IDs in the target cohort.")
    print(f"High exposure group contains {len(high_exposure_sample_ids)} samples.")
    print(f"Low exposure group contains {len(low_exposure_sample_ids)} samples.")

    sample_mutations, all_genes = process_maf_files(args, target_cohort_sample_ids)
    
    if not all_genes: # If no genes from MAFs, create empty output and exit.
        print("No mutated genes found in MAF files for the target cohort. GSEA analysis cannot proceed meaningfully.")
        # Create an empty .rnk file as per GSEA requirements for empty lists.
        try:
            save_ranked_list(pd.DataFrame(columns=['Gene', 'RankMetric']), args.output_ranked_gene_file)
        except ScriptLogicError: # Problem saving the file
             sys.exit(1)
        sys.exit(0) # Successful exit, but with empty results.

    mutation_matrix_df = create_binary_mutation_matrix(target_cohort_sample_ids, all_genes, sample_mutations)
    results_df = perform_differential_analysis(mutation_matrix_df, high_exposure_sample_ids, low_exposure_sample_ids, all_genes)
    
    try:
        save_ranked_list(results_df, args.output_ranked_gene_file)
    except ScriptLogicError:
        sys.exit(1)

if __name__ == "__main__":
    main()
