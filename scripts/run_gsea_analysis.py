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
    if prerank_obj is not None and hasattr(prerank_obj, 'results') and not prerank_obj.results.empty:
        print(f"GSEA analysis completed successfully. Results saved to: {args.output_dir_gsea}")
        # Optional: Could list top results or generate a quick plot here if desired
        # e.g., print(prerank_obj.results.head())
    else:
        msg = "GSEA analysis did not produce results or the result object was empty."
        print(f"Error: {msg}")
        # Consider raising a specific error if GSEApy can "succeed" but produce no results for valid reasons.
        # For now, treating as an error to be caught by main.
        raise Exception(msg)


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
