"""
Generates a ranked gene list based on regression analysis.

This script performs a regression analysis for each gene, modeling its mutation status
against signature exposure, TMB, and potentially other covariates. The output is a
ranked list of genes based on the significance (e.g., p-value) of the signature
exposure coefficient.
"""
import pandas as pd
import numpy as np
import argparse
import os
import glob
import sys
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import PerfectSeparationError, ConvergenceWarning
import gzip # For catching gzip.BadGzipFile

# Standardized sample ID column name to use internally after loading
SAMPLE_ID_COL = "Tumor_Sample_Barcode"

def create_parser():
    """Creates and returns the ArgumentParser object for the script."""
    parser = argparse.ArgumentParser(
        description="Generate a ranked gene list based on regression analysis of mutation status against signature exposure and TMB."
    )
    parser.add_argument(
        "--exposures_file", type=str, required=True,
        help="Path to the signature exposures file (TSV/CSV format). Expected to have a sample ID column/index."
    )
    parser.add_argument(
        "--sample_map_file", type=str, required=True,
        help="Path to the sample map file (TSV/CSV format) linking sample IDs to cohorts. Expected columns: 'SampleID' (or similar) and 'Cohort'."
    )
    parser.add_argument(
        "--tmb_file", type=str, required=True,
        help="Path to the TMB file (TSV/CSV format) containing TMB values per sample. Expected column: 'Tumor_Sample_Barcode' (or similar)."
    )
    parser.add_argument(
        "--maf_input_dir", type=str, required=True,
        help="Path to the root directory containing MAF files, organized by cohort subdirectories."
    )
    parser.add_argument(
        "--target_cohort", type=str, required=True,
        help="Target cohort ID for the analysis (e.g., 'TCGA-LUAD')."
    )
    parser.add_argument(
        "--target_signature_column", type=str, required=True,
        help="Name of the target signature column in the exposures file."
    )
    parser.add_argument(
        "--tmb_column_name", type=str, required=True,
        help="Name of the TMB column in the TMB file (e.g., 'TMB_mut_per_Mb' or 'Total_SNVs')."
    )
    parser.add_argument(
        "--min_mutations_per_gene", type=int, default=3,
        help="Minimum number of mutations required for a gene to be included in the analysis (default: 3)."
    )
    parser.add_argument(
        "--output_ranked_gene_file", type=str, required=True,
        help="Path to save the output ranked gene list file (TSV format)."
    )
    return parser

def _load_dataframe(file_path, file_type_name, expected_sample_col_options=None):
    """Helper to load a DataFrame and handle common errors."""
    print(f"Loading {file_type_name} file: {file_path}")
    try:
        df = pd.read_csv(file_path, sep=None, engine='python')
        if df.empty:
            print(f"Error: {file_type_name} file is empty: {file_path}", file=sys.stderr)
            sys.exit(1)
        print(f"  Loaded {file_type_name} data. Shape: {df.shape}")
        
        if expected_sample_col_options:
            found_col_name = None
            if df.index.name in expected_sample_col_options:
                df.index.name = SAMPLE_ID_COL
                found_col_name = SAMPLE_ID_COL 
            else: 
                for col_opt in expected_sample_col_options:
                    if col_opt in df.columns:
                        df.rename(columns={col_opt: SAMPLE_ID_COL}, inplace=True)
                        found_col_name = SAMPLE_ID_COL
                        break
            if not found_col_name:
                print(f"Error: Could not find one of the expected sample ID columns/index {expected_sample_col_options} in {file_type_name} file.", file=sys.stderr)
                sys.exit(1)
            if SAMPLE_ID_COL in df.columns and df.index.name != SAMPLE_ID_COL:
                df = df.set_index(SAMPLE_ID_COL)
            elif df.index.name != SAMPLE_ID_COL: 
                print(f"Error: Failed to set '{SAMPLE_ID_COL}' as index for {file_type_name} file.", file=sys.stderr)
                sys.exit(1)
        return df
    except FileNotFoundError:
        print(f"Error: {file_type_name} file not found: {file_path}", file=sys.stderr); sys.exit(1)
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
        print(f"Error parsing {file_type_name} file {file_path}: {e}. Ensure it's a valid CSV or TSV file.", file=sys.stderr); sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred loading {file_type_name} file {file_path}: {e}", file=sys.stderr); sys.exit(1)

def load_and_prepare_metadata(args):
    exposures_df = _load_dataframe(args.exposures_file, "Exposures", expected_sample_col_options=['Tumor_Sample_Barcode', 'SampleID', 'sample_id'])
    sample_map_df = _load_dataframe(args.sample_map_file, "Sample Map", expected_sample_col_options=['SampleID', 'Tumor_Sample_Barcode'])
    if 'Cohort' not in sample_map_df.columns:
        print(f"Error: 'Cohort' column not found in sample map file: {args.sample_map_file}", file=sys.stderr); sys.exit(1)
    tmb_df = _load_dataframe(args.tmb_file, "TMB", expected_sample_col_options=['Tumor_Sample_Barcode', 'SampleID'])
    temp_tmb_df_cols_and_index = list(tmb_df.columns) + ([tmb_df.index.name] if tmb_df.index.name else [])
    if args.tmb_column_name not in temp_tmb_df_cols_and_index:
        print(f"Error: TMB column '{args.tmb_column_name}' not found in TMB file: {args.tmb_file}. Available: {temp_tmb_df_cols_and_index}", file=sys.stderr); sys.exit(1)

    print(f"\nMerging DataFrames on '{SAMPLE_ID_COL}' (index)...")
    merged_df = pd.merge(exposures_df, sample_map_df, left_index=True, right_index=True, how='inner')
    if merged_df.empty: print("Error: Merging exposures and sample map resulted in an empty DataFrame.", file=sys.stderr); sys.exit(1)
    print(f"  Shape after merging exposures and sample map: {merged_df.shape}")
    
    final_merged_df = pd.merge(merged_df, tmb_df[[args.tmb_column_name]], left_index=True, right_index=True, how='inner')
    if final_merged_df.empty: print("Error: Merging with TMB data resulted in an empty DataFrame.", file=sys.stderr); sys.exit(1)
    print(f"  Shape after merging with TMB data: {final_merged_df.shape}")

    print(f"\nFiltering for target cohort: {args.target_cohort}")
    cohort_analysis_df = final_merged_df[final_merged_df['Cohort'] == args.target_cohort]
    if cohort_analysis_df.empty:
        print(f"Error: No samples found for target cohort '{args.target_cohort}'.", file=sys.stderr); sys.exit(1)
    print(f"  Found {len(cohort_analysis_df)} samples for cohort '{args.target_cohort}'. Shape: {cohort_analysis_df.shape}")

    missing_final_cols = [col for col in [args.target_signature_column, args.tmb_column_name] if col not in cohort_analysis_df.columns]
    if missing_final_cols:
        print(f"Error: Required columns missing: {', '.join(missing_final_cols)}. Available: {cohort_analysis_df.columns.tolist()}", file=sys.stderr); sys.exit(1)
    print(f"Final cohort analysis DataFrame prepared. Index: '{cohort_analysis_df.index.name}'. Shape: {cohort_analysis_df.shape}")
    return cohort_analysis_df

def build_binary_mutation_matrix(args, cohort_metadata_df):
    target_sample_barcodes = list(cohort_metadata_df.index)
    if not target_sample_barcodes:
        print("Warning: No target samples for mutation matrix.", file=sys.stderr); return pd.DataFrame()
    print(f"\nBuilding binary mutation matrix for {len(target_sample_barcodes)} samples in cohort '{args.target_cohort}'...")
    sample_mutations = {sample_id: set() for sample_id in target_sample_barcodes}
    all_gene_symbols_in_cohort = set()
    cohort_maf_dir = os.path.join(args.maf_input_dir, args.target_cohort)
    if not os.path.isdir(cohort_maf_dir):
        print(f"  Warning: MAF dir not found for {args.target_cohort} at {cohort_maf_dir}. Matrix will be empty of mutations.", file=sys.stderr)

    all_maf_files = glob.glob(os.path.join(cohort_maf_dir, '*.maf.gz')) + glob.glob(os.path.join(cohort_maf_dir, '*.maf'))
    if not all_maf_files: print(f"  Warning: No MAF files found for {args.target_cohort}. Matrix will be empty of mutations.", file=sys.stderr)
    else:
        print(f"  Found {len(all_maf_files)} MAF file(s) for cohort {args.target_cohort}.")
        required_maf_cols = ['Tumor_Sample_Barcode', 'Variant_Type', 'Reference_Allele', 'Tumor_Seq_Allele2', 'Hugo_Symbol']
        for maf_file_path in all_maf_files:
            # print(f"    Processing MAF file for mutation matrix: {maf_file_path}") # Can be verbose
            try:
                maf_df = pd.read_csv(maf_file_path, sep='\t', comment='#', low_memory=False, compression=('gzip' if maf_file_path.endswith('.gz') else None), usecols=lambda c: c in required_maf_cols)
            except ValueError as ve: print(f"      Error: Essential columns missing in {maf_file_path} (or issue with usecols). {ve}", file=sys.stderr); continue
            except Exception as e: print(f"      Error reading MAF {maf_file_path}: {e}", file=sys.stderr); continue
            
            if any(col not in maf_df.columns for col in required_maf_cols): print(f"      Error: MAF {maf_file_path} missing columns after load. Skipping.", file=sys.stderr); continue
            
            maf_df_filtered_samples = maf_df[maf_df[SAMPLE_ID_COL].isin(target_sample_barcodes)]
            if maf_df_filtered_samples.empty: continue
            snp_df = maf_df_filtered_samples[maf_df_filtered_samples['Variant_Type'] == 'SNP']
            if snp_df.empty: continue
            true_snvs_df = snp_df[snp_df['Reference_Allele'].astype(str).str.match(r'^[ACGT]$') & snp_df['Tumor_Seq_Allele2'].astype(str).str.match(r'^[ACGT]$') & (snp_df['Reference_Allele'] != snp_df['Tumor_Seq_Allele2'])]
            for _, row in true_snvs_df.iterrows():
                sample_mutations[row[SAMPLE_ID_COL]].add(row['Hugo_Symbol']); all_gene_symbols_in_cohort.add(row['Hugo_Symbol'])
    
    if not all_gene_symbols_in_cohort:
        print(f"Warning: No mutations found for any gene in target cohort MAFs for '{args.target_cohort}'. Returning matrix of zeros.", file=sys.stderr)
        return pd.DataFrame(0, index=target_sample_barcodes, columns=[]).astype(int) 
    
    sorted_genes = sorted(list(all_gene_symbols_in_cohort))
    binary_mutation_matrix = pd.DataFrame(0, index=target_sample_barcodes, columns=sorted_genes)
    for sample_id, mutated_genes in sample_mutations.items():
        if sample_id in binary_mutation_matrix.index:
            for gene in mutated_genes:
                if gene in binary_mutation_matrix.columns: binary_mutation_matrix.loc[sample_id, gene] = 1
    binary_mutation_matrix = binary_mutation_matrix.fillna(0).astype(int) 
    print(f"  Binary mutation matrix created. Shape: {binary_mutation_matrix.shape}")
    return binary_mutation_matrix

def perform_gene_wise_regression(cohort_metadata_df, binary_mutation_matrix, args):
    results_list = []
    genes_to_process = binary_mutation_matrix.columns
    num_total_genes = len(genes_to_process)
    if num_total_genes == 0:
        print("Warning: No genes in binary mutation matrix to process for regression.", file=sys.stderr)
        return results_list
    print(f"\n--- Starting Gene-wise Logistic Regression for {num_total_genes} genes ---")
    processed_count, skipped_low_mutation_count, failed_model_fit_count = 0, 0, 0

    for i, gene in enumerate(genes_to_process):
        if (i + 1) % 50 == 0 or (i + 1) == num_total_genes : 
            print(f"  Processing gene {i+1} of {num_total_genes}: {gene}...")
        num_mutations = binary_mutation_matrix[gene].sum()
        if num_mutations < args.min_mutations_per_gene:
            skipped_low_mutation_count +=1; continue
        
        Y = binary_mutation_matrix[gene].astype(float)
        common_index = cohort_metadata_df.index.intersection(Y.index)
        Y_aligned = Y.loc[common_index]
        X_df_aligned = cohort_metadata_df.loc[common_index, [args.target_signature_column, args.tmb_column_name]].copy()
        
        if X_df_aligned.isnull().values.any():
            X_df_aligned.dropna(inplace=True); Y_aligned = Y_aligned.loc[X_df_aligned.index]
        if len(Y_aligned) < args.min_mutations_per_gene + 2: # Heuristic for enough data points
            skipped_low_mutation_count +=1; continue
            
        X_df_const = sm.add_constant(X_df_aligned, prepend=True)
        try:
            model = sm.Logit(Y_aligned, X_df_const); result = model.fit(disp=0)
            coeff_sig = result.params.get(args.target_signature_column)
            pval_sig = result.pvalues.get(args.target_signature_column)
            z_score_sig = result.tvalues.get(args.target_signature_column)
            if coeff_sig is None or pval_sig is None or z_score_sig is None:
                raise ValueError("Missing regression results for signature column.")
            pval_capped = max(float(pval_sig), np.nextafter(0,1))
            rank_metric = np.sign(float(coeff_sig)) * -np.log10(pval_capped)
            results_list.append({'Gene': gene, 'Coefficient_Signature': float(coeff_sig), 'PValue_Signature': float(pval_sig), 
                                 'Z_Score_Signature': float(z_score_sig), 'RankMetric': rank_metric, 
                                 'Num_Mutations': num_mutations, 'Status': 'Success'})
            processed_count += 1
        except (np.linalg.LinAlgError, PerfectSeparationError, ConvergenceWarning, ValueError, Exception) as e:
            failed_model_fit_count +=1
            results_list.append({'Gene': gene, 'Coefficient_Signature': np.nan, 'PValue_Signature': np.nan, 
                                 'Z_Score_Signature': np.nan, 'RankMetric': 0.0, 
                                 'Num_Mutations': num_mutations, 'Status': f'Failed_Fit_{type(e).__name__}'})
    print(f"--- Finished Gene-wise Logistic Regression ---")
    print(f"  Total genes considered: {num_total_genes}\n  Successfully processed: {processed_count}\n"
          f"  Skipped (low mutations/data): {skipped_low_mutation_count}\n  Model fit failures: {failed_model_fit_count}")
    return results_list

def main(args_list=None):
    parser = create_parser()
    try: args = parser.parse_args(args_list if args_list is not None else sys.argv[1:])
    except SystemExit as e: sys.exit(2) 

    print("Parsed arguments:")
    for arg, value in vars(args).items(): print(f"  {arg.replace('_', ' ').title()}: {value}")

    print("\n--- Starting Data Loading and Preparation ---")
    metadata_df = load_and_prepare_metadata(args)
    print("--- Finished Data Loading and Preparation ---")
    if metadata_df.empty: print("Error: Metadata DataFrame is empty. Exiting.", file=sys.stderr); sys.exit(1)
    print(f"Shape of metadata_df for regression: {metadata_df.shape}")
    
    print("\n--- Starting Binary Mutation Matrix Construction ---")
    mutation_matrix_df = build_binary_mutation_matrix(args, metadata_df)
    print("--- Finished Binary Mutation Matrix Construction ---")
    if mutation_matrix_df.shape[1] == 0 and mutation_matrix_df.shape[0] > 0 :
         print("Warning: Mutation matrix contains no genes. Regression step will be skipped, and an empty ranked list will be produced.", file=sys.stderr)
    elif mutation_matrix_df.empty: 
         print("Error: Mutation matrix is completely empty. Cannot proceed.", file=sys.stderr)
         sys.exit(1)
    print(f"Mutation matrix shape: {mutation_matrix_df.shape}")

    regression_results_list = perform_gene_wise_regression(metadata_df, mutation_matrix_df, args)

    print("\n--- Starting Output File Generation ---")
    if not regression_results_list: # Handles empty list from no genes or all failing
        print("No regression results to save. Output file will not be created.")
        print("\nScript finished.")
        return # Exit gracefully

    results_df = pd.DataFrame(regression_results_list)
    
    # Optional: Save full diagnostic results
    full_results_output_path = args.output_ranked_gene_file.replace(".tsv", "_full_stats.tsv")
    try:
        output_dir_full = os.path.dirname(full_results_output_path)
        if output_dir_full and not os.path.exists(output_dir_full):
            os.makedirs(output_dir_full, exist_ok=True)
        results_df.to_csv(full_results_output_path, sep='\t', index=False, na_rep='NaN')
        print(f"  Full diagnostic results saved to: {full_results_output_path}")
    except Exception as e:
        print(f"  Warning: Could not save full diagnostic results to {full_results_output_path}. Error: {e}", file=sys.stderr)

    # Prepare and save the GSEA .rnk style ranked list
    results_df_sorted = results_df.sort_values(by='RankMetric', ascending=False)
    output_df = results_df_sorted[['Gene', 'RankMetric']]
    
    try:
        output_dir_ranked = os.path.dirname(args.output_ranked_gene_file)
        if output_dir_ranked and not os.path.exists(output_dir_ranked):
            # This might be redundant if full_results_output_path has the same dir
            os.makedirs(output_dir_ranked, exist_ok=True) 
            print(f"  Created output directory (for ranked list): {output_dir_ranked}") # Log only if created here
        
        output_df.to_csv(args.output_ranked_gene_file, sep='\t', index=False, header=False, na_rep='NaN')
        print(f"  Ranked gene list saved to: {args.output_ranked_gene_file}")
        print(f"  Number of genes in ranked list: {len(output_df)}")
    except Exception as e:
        print(f"Error saving ranked gene list to {args.output_ranked_gene_file}: {e}", file=sys.stderr)
        sys.exit(1)
    print("--- Finished Output File Generation ---")
    
    print("\nScript finished successfully.")

if __name__ == "__main__":
    main()
