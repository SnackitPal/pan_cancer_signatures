"""
Generates a summary bar plot from GSEA Prerank results.
This script takes a GSEA Prerank result file, filters and sorts pathways,
and generates a bar plot of the top N pathways, highlighting significant ones.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import sys
import numpy as np # Added for isnumeric check if needed, and for abs

def create_parser():
    """Creates and returns the ArgumentParser object for the script."""
    parser = argparse.ArgumentParser(description="Generate a summary bar plot from GSEA Prerank results.")
    parser.add_argument("--gsea_results_file", type=str, required=True,
                        help="Path to GSEA Prerank results file (e.g., 'gseapy.prerank.gene_sets.report.csv').")
    parser.add_argument("--top_n_pathways", type=int, default=20,
                        help="Number of top pathways to display (default: 20).")
    parser.add_argument("--output_plot_file", type=str, required=True,
                        help="Path to save output plot image (e.g., 'gsea_summary.png').")
    parser.add_argument("--sort_by_metric", type=str, default='NES',
                        help="Metric to sort pathways by (e.g., 'NES', 'NOM p-val', 'FDR q-val'). Default: 'NES'.")
    parser.add_argument("--fdr_threshold", type=float, default=0.25,
                        help="FDR q-value threshold for highlighting. Default: 0.25.")
    return parser

def load_and_validate_gsea_data(gsea_results_file_path, expected_sort_metric_col):
    """
    Loads GSEA results, validates columns, and filters NaNs. Returns DataFrame.
    Raises FileNotFoundError, ValueError for issues.
    """
    print(f"Loading GSEA results file: {gsea_results_file_path}")
    try:
        gsea_df = pd.read_csv(gsea_results_file_path, sep=None, engine='python')
        if gsea_df.empty: raise ValueError("GSEA results file is empty.")
    except FileNotFoundError:
        print(f"Error: GSEA results file not found at {gsea_results_file_path}"); raise
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
        print(f"Error parsing GSEA results file {gsea_results_file_path}: {e}"); raise ValueError(f"Parse error: {e}")
    except Exception as e:
        print(f"Unexpected error loading GSEA results {gsea_results_file_path}: {e}"); raise

    print("Validating columns...")
    essential_cols = ['Term', 'NES', 'FDR q-val']
    if expected_sort_metric_col not in essential_cols: essential_cols.append(expected_sort_metric_col)
    
    missing_cols = [col for col in essential_cols if col not in gsea_df.columns]
    if missing_cols:
        msg = f"Missing essential columns: {', '.join(missing_cols)}. Available: {', '.join(gsea_df.columns)}"
        print(f"Error: {msg}"); raise ValueError(msg)
    print(f"Essential columns found: {', '.join(essential_cols)}.")

    print("Filtering NaNs in 'NES' or 'FDR q-val'...")
    gsea_df.dropna(subset=['NES', 'FDR q-val'], inplace=True)
    if gsea_df.empty:
        msg = "No data after NaN filter for NES/FDR. Cannot proceed."
        print(f"Warning: {msg}"); raise ValueError(msg)
    print(f"Data shape after NaN filter: {gsea_df.shape}")
    return gsea_df

def select_top_pathways(df_validated, top_n, sort_by_metric_col):
    """
    Selects and sorts top N pathways based on the sort_by_metric.
    Returns a DataFrame ready for plotting.
    Raises ValueError if issues occur (e.g., sort_by_metric not numeric, or no pathways selected).
    """
    print(f"Selecting top {top_n} pathways by abs value of '{sort_by_metric_col}'...")
    if top_n == 0:
        print("Warning: top_n_pathways is 0, no pathways will be selected for plotting.")
        return pd.DataFrame(columns=df_validated.columns) # Return empty DF with same columns

    if not pd.api.types.is_numeric_dtype(df_validated[sort_by_metric_col]):
        msg = f"Column '{sort_by_metric_col}' for sorting is not numeric."
        print(f"Error: {msg}"); raise ValueError(msg)
        
    df_validated['abs_sort_metric'] = df_validated[sort_by_metric_col].abs()
    
    # Handle cases where there are fewer pathways than top_n requested
    actual_top_n = min(top_n, len(df_validated))
    top_n_df = df_validated.sort_values(by='abs_sort_metric', ascending=False).head(actual_top_n)
    
    if top_n_df.empty and top_n > 0 : # Check if top_n was > 0 because if top_n is 0, empty is expected.
        msg = (f"No pathways selected (top {top_n}). Filtered GSEA results ({len(df_validated)} pathways) "
               "might be empty or top_n is too small.")
        print(msg); raise ValueError(msg)

    # Sort these top N pathways by their original 'NES' for plotting
    plot_df = top_n_df.sort_values(by='NES', ascending=False)
    print(f"Selected {len(plot_df)} pathways, sorted by 'NES' for display.")
    return plot_df

def generate_plot_and_save(df_to_plot, output_plot_file_path, fdr_threshold_for_highlighting, num_selected_pathways_for_title):
    """
    Generates the GSEA summary bar plot and saves it.
    df_to_plot should be pre-sorted and pre-filtered.
    num_selected_pathways_for_title is used for the plot title.
    Raises Exception for plotting/saving errors.
    """
    if df_to_plot.empty:
        print("Warning: No data for plotting. Skipping plot generation.")
        # Create an empty file or a file with a message? For now, just skip.
        # To ensure a file is created as per some expectations, one might do:
        # with open(output_plot_file_path, 'w') as f: f.write("No data to plot.")
        return

    print(f"\nGenerating plot for top {len(df_to_plot)} pathways...")
    output_plot_dir = os.path.dirname(output_plot_file_path)
    if output_plot_dir and not os.path.exists(output_plot_dir):
        try:
            os.makedirs(output_plot_dir, exist_ok=True)
            print(f"Created output directory: {output_plot_dir}")
        except Exception as e:
            print(f"Error creating directory {output_plot_dir}: {e}"); raise

    df_to_plot['Term_Display'] = df_to_plot.apply(
        lambda r: r['Term'] + ' *' if r['FDR q-val'] < fdr_threshold_for_highlighting else r['Term'], axis=1
    )

    original_fdr_col_name = 'FDR q-val'
    try:
        nes_col_tuple_idx = df_to_plot.columns.get_loc('NES') + 1
        fdr_col_tuple_idx = df_to_plot.columns.get_loc(original_fdr_col_name) + 1
    except KeyError as e:
        print(f"Error: A required column ('NES' or '{original_fdr_col_name}') not found in DataFrame for plotting. {e}")
        print(f"Available columns: {df_to_plot.columns.tolist()}")
        # Decide how to handle this - maybe return or raise a specific error
        return 

    plt.style.use('ggplot') 
    fig, ax = plt.subplots(figsize=(12, max(6, len(df_to_plot) * 0.45))) 
    palette = {True: 'firebrick', False: 'cornflowerblue'}
    bar_colors = [palette[nes >= 0] for nes in df_to_plot['NES']]
    sns.barplot(x='NES', y='Term_Display', data=df_to_plot, palette=bar_colors, ax=ax, orient='h')


    for i, row_tuple in enumerate(df_to_plot.itertuples()): # Use itertuples for efficiency and direct attribute access
        nes_val = row_tuple[nes_col_tuple_idx] 
        fdr_val = row_tuple[fdr_col_tuple_idx] # Access column with space in name

        ha = 'left' if nes_val >= 0 else 'right'
        x_offset_factor = 0.05
        # Use current axis limits to make offset relative
        current_xlim = ax.get_xlim()
        x_offset = x_offset_factor * (current_xlim[1] if nes_val >=0 else abs(current_xlim[0]))
        text_x = nes_val + x_offset if nes_val >=0 else nes_val - x_offset

        if abs(nes_val) < (0.1 * (current_xlim[1] - current_xlim[0])): # If bar is very short
             text_x = x_offset if nes_val >=0 else -x_offset
        
        ax.text(text_x, i, f"FDR: {fdr_val:.2g}", va='center', ha=ha, fontsize=9, color='black')

    ax.set_title(f"Top {num_selected_pathways_for_title} Enriched Pathways by NES", fontsize=16)
    ax.set_xlabel("Normalized Enrichment Score (NES)", fontsize=12)
    ax.set_ylabel("Pathway", fontsize=12)
    ax.axvline(0, color='black', lw=1.0, linestyle='--')
    ax.grid(True, axis='x', linestyle=':', alpha=0.7); ax.grid(False, axis='y') 
    plt.yticks(fontsize=10); plt.tight_layout(pad=1.5)

    try:
        plt.savefig(output_plot_file_path, dpi=300, bbox_inches='tight')
        print(f"Summary plot saved to: {output_plot_file_path}")
    except Exception as e:
        print(f"Error saving plot to {output_plot_file_path}: {e}"); raise
    finally:
        plt.close(fig)

def plot_gsea_summary(args):
    """
    Main orchestrator for loading, selecting, and plotting GSEA summary.
    """
    df_validated = load_and_validate_gsea_data(args.gsea_results_file, args.sort_by_metric)
    df_selected_for_plot = select_top_pathways(df_validated, args.top_n_pathways, args.sort_by_metric)
    
    # Pass len(df_selected_for_plot) for title, as top_n_pathways might be > actual pathways found
    generate_plot_and_save(df_selected_for_plot, args.output_plot_file, args.fdr_threshold, len(df_selected_for_plot))

def main(args_list=None):
    """Main function to parse arguments and call the plotting function."""
    parser = create_parser()
    args = parser.parse_args(args_list if args_list is not None else sys.argv[1:])
    
    if args.top_n_pathways < 0: # Allow 0 for no pathways, but not negative
        print("Error: --top_n_pathways must be a non-negative integer.")
        sys.exit(1)

    try:
        plot_gsea_summary(args)
        print("\nScript finished successfully.")
    except FileNotFoundError:
        print("Exiting due to file not found."); sys.exit(1)
    except ValueError as ve:
        print(f"Exiting due to a value or data error: {ve}"); sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}"); sys.exit(1)

if __name__ == "__main__":
    main()
