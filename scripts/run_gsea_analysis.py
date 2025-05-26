import gseapy
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import sys # Added for potential future use, good practice

def create_parser():
    """Creates and returns the ArgumentParser object."""
    parser = argparse.ArgumentParser(description="Run GSEA analysis using gseapy.")
    parser.add_argument("--ranked_gene_list_file", type=str, required=True,
                        help="Path to the pre-ranked gene list file (e.g., .rnk format).")
    parser.add_argument("--gene_sets_gmt_file", type=str, required=True,
                        help="Path to the gene sets GMT file (e.g., from MSigDB).")
    parser.add_argument("--output_dir_gsea", type=str, required=True,
                        help="Directory to save the GSEA results.")
    parser.add_argument("--min_gene_set_size", type=int, default=15,
                        help="Minimum size of gene sets to include in the analysis.")
    parser.add_argument("--max_gene_set_size", type=int, default=500,
                        help="Maximum size of gene sets to include in the analysis.")
    parser.add_argument("--permutation_num", type=int, default=1000,
                        help="Number of permutations for the GSEA analysis.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    return parser

def run_gsea(args):
    """
    Placeholder function for running the GSEA analysis.
    Currently, it just prints the parsed arguments.
    """
    """
    Runs the GSEA Prerank analysis using gseapy.
    """
    # 1. Load Ranked Gene List
    print(f"Loading ranked gene list file: {args.ranked_gene_list_file}")
    try:
        ranked_df = pd.read_csv(args.ranked_gene_list_file, sep='\t', header=None, names=['gene_symbol', 'rank_metric'])
        if ranked_df.shape[1] != 2:
            raise ValueError("Ranked gene list file must have exactly two columns (Gene Symbol, Rank Metric).")
        if ranked_df.isnull().values.any():
             raise ValueError("Ranked gene list file contains missing values.")
        if not pd.api.types.is_numeric_dtype(ranked_df['rank_metric']):
            raise ValueError("Rank metric column in ranked gene list file must be numeric.")

        # Convert to Series as preferred by gseapy for rnk input
        rank_series = ranked_df.set_index('gene_symbol')['rank_metric']
        print(f"Successfully loaded and parsed ranked gene list. Found {len(rank_series)} genes.")

    except FileNotFoundError:
        print(f"Error: Ranked gene list file not found at {args.ranked_gene_list_file}")
        raise # Re-raise to be caught by main
    except ValueError as ve:
        print(f"Error parsing ranked gene list file {args.ranked_gene_list_file}: {ve}")
        raise # Re-raise to be caught by main
    except Exception as e:
        print(f"An unexpected error occurred while loading ranked gene list {args.ranked_gene_list_file}: {e}")
        raise # Re-raise to be caught by main

    # 2. Create Output Directory
    print(f"Creating output directory: {args.output_dir_gsea}")
    try:
        os.makedirs(args.output_dir_gsea, exist_ok=True)
    except Exception as e:
        print(f"Error creating output directory {args.output_dir_gsea}: {e}")
        raise # Re-raise to be caught by main

    # 3. Run GSEA Preranked Analysis
    print("Starting GSEA Prerank analysis...")
    try:
        # Ensure gene_sets file exists before calling gseapy, as it might not give a clear FileNotFoundError
        if not os.path.exists(args.gene_sets_gmt_file):
            raise FileNotFoundError(f"Gene sets GMT file not found: {args.gene_sets_gmt_file}")

        prerank_obj = gseapy.prerank(
            rnk=rank_series,
            gene_sets=args.gene_sets_gmt_file,
            outdir=args.output_dir_gsea,
            min_size=args.min_gene_set_size,
            max_size=args.max_gene_set_size,
            permutation_num=args.permutation_num,
            seed=args.seed,
            verbose=True
        )
    except FileNotFoundError as e: # Specifically for the GMT file check
        print(f"Error: {e}")
        raise
    except Exception as e:
        print(f"Error during GSEApy Prerank execution: {e}")
        # This could be due to various issues: bad GMT format, issues with gene names, internal GSEApy errors.
        raise # Re-raise to be caught by main

    # 4. Confirmation and Logging
    print("GSEApy Prerank function call completed.") # Indicates gseapy itself didn't crash

    # GSEApy, when an outdir is specified, writes its primary results to files.
    # The most common primary results table filename pattern is 'gseapy.gsea.prerank.gene_sets.report.csv'
    # or 'gseapy.prerank.gene_sets.report.csv'. It can also be .tsv.
    # We will check for the existence and non-emptiness of such a file.

    # List potential report file names (GSEApy's naming can sometimes vary slightly or be configured)
    potential_report_files = [
        "gseapy.gsea.prerank.gene_sets.report.csv",
        "gseapy.gsea.prerank.gene_sets.report.tsv",
        "gseapy.prerank.gene_sets.report.csv", # Older or different gseapy versions/calls
        "gseapy.prerank.gene_sets.report.tsv"
    ]
    
    found_report_file = None
    for report_file_name in potential_report_files:
        expected_results_path = os.path.join(args.output_dir_gsea, report_file_name)
        if os.path.exists(expected_results_path):
            found_report_file = expected_results_path
            break
    
    if found_report_file:
        print(f"Found potential GSEA results table: {found_report_file}")
        try:
            results_df_check = pd.read_csv(found_report_file, sep='\t' if found_report_file.endswith('.tsv') else ',')
            if not results_df_check.empty:
                print(f"GSEA analysis appears successful. Main results table loaded and is not empty.")
                print(f"Full results saved to directory: {args.output_dir_gsea}")
                # You could print a snippet of the results if desired:
                # print("\nTop results from GSEA table:")
                # print(results_df_check.head())
            else:
                # GSEApy ran, file exists, but it's empty - this is a warning condition.
                print(f"Warning: GSEA analysis completed, but the main results file ({found_report_file}) is empty.")
                print(f"Full GSEApy output is in directory: {args.output_dir_gsea}")
        except Exception as e_load:
            # GSEApy ran, file exists, but we couldn't load it - also a warning.
            print(f"Warning: GSEA analysis completed, but could not load or verify the main results file {found_report_file}. Error: {e_load}")
            print(f"Full GSEApy output is in directory: {args.output_dir_gsea}")
    else:
        # GSEApy Prerank function completed without an error, but no standard output report file was found.
        # This might indicate an issue or a different GSEApy configuration/version.
        print("Warning: GSEApy Prerank function call completed, but a standard GSEA results report file was not found in the output directory.")
        print(f"Please manually check the contents of {args.output_dir_gsea} for GSEA results.")
        # We don't raise an Exception here because GSEApy itself didn't error out.
        # The user should inspect the output directory.

# The main function that calls run_gsea would remain largely the same,
# as run_gsea will now raise exceptions for critical failures (like GSEApy crashing or GMT not found)
# but will only print warnings if the output file check is ambiguous.


def main(args_list=None): # args_list for testing flexibility
    """Main function to parse arguments and call GSEA execution."""
    parser = create_parser()
    args = parser.parse_args(args_list if args_list is not None else sys.argv[1:])
    
    try:
        run_gsea(args)
        print("Script finished successfully.")
    except FileNotFoundError: # Catch only FileNotFoundError if we want specific exit codes later
        # Error message already printed by run_gsea or its sub-functions
        print("Exiting due to file not found.")
        sys.exit(1)
    except ValueError: # Catch only ValueError for parsing issues
        print("Exiting due to data parsing error.")
        sys.exit(1)
    except Exception as e:
        # General catch-all for other exceptions from run_gsea (incl. GSEApy errors)
        print(f"An unhandled error occurred during GSEA execution: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
