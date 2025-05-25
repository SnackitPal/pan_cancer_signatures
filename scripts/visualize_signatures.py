import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

# Attempt to import from signature_plotting_utils.
# This assumes signature_plotting_utils.py is in the same directory (scripts/)
# or accessible via PYTHONPATH.
try:
    from signature_plotting_utils import MUTATION_TYPE_COLORS, STANDARD_96_CONTEXTS, get_reordered_contexts
except ImportError:
    print("Error: Could not import from signature_plotting_utils.py.")
    print("Ensure the file is in the same directory (scripts/) or PYTHONPATH is set correctly.")
    # As a fallback for basic script structure to work, define them if not imported.
    # This is not ideal for production but helps in isolated testing of this script's structure.
    if 'MUTATION_TYPE_COLORS' not in globals():
        MUTATION_TYPE_COLORS = {'C>A': 'blue', 'C>G': 'black', 'C>T': 'red', 
                                'T>A': 'grey', 'T>C': 'green', 'T>G': 'pink'}
    if 'STANDARD_96_CONTEXTS' not in globals():
        # Generate a placeholder if not available, real one is complex
        STANDARD_96_CONTEXTS = [f"Ctx{i}" for i in range(96)] 
    if 'get_reordered_contexts' not in globals():
        def get_reordered_contexts(cols): return sorted(list(set(cols) & set(STANDARD_96_CONTEXTS)))

def main():
    parser = argparse.ArgumentParser(description="Visualize mutational signature profiles.")
    parser.add_argument("--signature_profiles_path", type=str, required=True,
                        help="Path to the input signature profiles CSV file.")
    parser.add_argument("--output_dir_figures", type=str, required=True,
                        help="Path to the directory where signature plots will be saved.")
    parser.add_argument("--file_prefix", type=str, default="signature_",
                        help="Prefix for output plot filenames (default: 'signature_').")
    
    args = parser.parse_args()

    # Print input parameters
    print("Starting signature visualization script with the following parameters:")
    print(f"  Signature profiles path: {args.signature_profiles_path}")
    print(f"  Output directory for figures: {args.output_dir_figures}")
    print(f"  File prefix for plots: {args.file_prefix}")

    # Data Loading and Preparation
    print(f"\nLoading signature profiles from: {args.signature_profiles_path}")
    try:
        # The first column of the CSV (e.g., 'Signature_1') should become the index.
        df_profiles = pd.read_csv(args.signature_profiles_path, index_col=0)
    except FileNotFoundError:
        print(f"Error: Input signature profiles file not found at {args.signature_profiles_path}")
        return # Exit script
    except Exception as e:
        print(f"Error loading CSV file {args.signature_profiles_path}: {e}")
        return

    if df_profiles.empty:
        print(f"Error: Signature profiles file at {args.signature_profiles_path} is empty.")
        return

    # Reorder Columns for plotting
    print("Reordering contexts for plotting...")
    loaded_contexts = df_profiles.columns.tolist()
    
    # Check if signature_plotting_utils was successfully imported to use its functions
    if 'get_reordered_contexts' in globals() and callable(get_reordered_contexts):
        plot_ordered_contexts = get_reordered_contexts(loaded_contexts)
    else:
        # Fallback if import failed - this won't be the desired order
        print("Warning: Using basic sort for contexts due to import issue. Plot order may not be standard.")
        plot_ordered_contexts = sorted(loaded_contexts) 

    df_profiles = df_profiles[plot_ordered_contexts]

    print(f"Loaded and reordered signature profiles with shape: {df_profiles.shape}")

    # Validate the number of columns (contexts)
    # Using len(STANDARD_96_CONTEXTS) is more robust if STANDARD_96_CONTEXTS could be a subset
    # (though typically it's fixed at 96).
    expected_num_contexts = len(STANDARD_96_CONTEXTS) if 'STANDARD_96_CONTEXTS' in globals() and STANDARD_96_CONTEXTS else 96
    
    if df_profiles.shape[1] != expected_num_contexts:
        # This check is against the potentially placeholder STANDARD_96_CONTEXTS if import failed.
        # A more accurate check is against the *actual* length of the contexts expected for plotting.
        # If get_reordered_contexts worked, plot_ordered_contexts holds the intersection.
        # The warning should be if plot_ordered_contexts is not 96.
        if len(plot_ordered_contexts) != 96 :
             print(f"Warning: The reordered profiles matrix has {len(plot_ordered_contexts)} columns. "
                   f"Standard plotting expects 96 contexts. Some contexts might be missing from input or not in standard list.")
        elif df_profiles.shape[1] != 96 : # If plot_ordered_contexts is 96 but original wasn't (e.g. extra cols dropped)
             print(f"Info: The loaded profiles matrix initially had {len(loaded_contexts)} columns, "
                   f"which was reordered to {len(plot_ordered_contexts)} standard contexts for plotting.")


    # Signature plotting will go here
    print("\nStarting signature plotting...")

    # Create output directory for figures
    try:
        os.makedirs(args.output_dir_figures, exist_ok=True)
        print(f"Output directory for figures: {args.output_dir_figures}")
    except OSError as e:
        print(f"Error creating output directory {args.output_dir_figures}: {e}")
        return # Exit if directory cannot be created

    # Iterate through each signature (row in df_profiles) and plot
    for signature_name, signature_data in df_profiles.iterrows():
        plot_title = f"{args.file_prefix}{signature_name}"
        # Ensure filename is valid (e.g. replace spaces, etc. if prefix or name can have them)
        safe_signature_name = str(signature_name).replace(" ", "_").replace("/", "_")
        output_filename = f"{args.file_prefix}{safe_signature_name}.png"
        full_output_path = os.path.join(args.output_dir_figures, output_filename)

        print(f"  Plotting {plot_title}...")
        try:
            plot_signature(
                signature_series=signature_data,
                signature_name=plot_title, # Use the constructed plot_title
                output_path=full_output_path,
                colors_map=MUTATION_TYPE_COLORS,
                ordered_contexts=plot_ordered_contexts, # Use the reordered contexts from Part 1
                mutation_types_order=MUTATION_TYPES # from plotting utils
            )
            print(f"    Saved plot to {full_output_path}")
        except Exception as e:
            print(f"    Error plotting or saving signature {signature_name}: {e}")
            # Optionally, continue to next signature or re-raise

    print("\nAll signature plots processed.")


# Define the plotting function
def plot_signature(signature_series, signature_name, output_path, colors_map, ordered_contexts, mutation_types_order):
    """
    Plots a single mutational signature profile.

    Args:
        signature_series (pd.Series): A pandas Series with 96 context probabilities for one signature.
        signature_name (str): The name of the signature (for the plot title).
        output_path (str): Full path to save the plot image.
        colors_map (dict): Dictionary mapping mutation types (e.g., 'C>A') to colors.
        ordered_contexts (list): List of the 96 trinucleotide contexts in standard plotting order.
        mutation_types_order (list): List of the 6 base mutation types (e.g. 'C>A') in plotting order.
    """
    if not (isinstance(ordered_contexts, list) and len(ordered_contexts) == 96):
        print(f"Warning for {signature_name}: ordered_contexts is not a list of 96 items. Using series index if available.")
        if isinstance(signature_series.index, pd.Index) and len(signature_series.index) == 96:
            ordered_contexts = signature_series.index.tolist()
        else:
            raise ValueError("ordered_contexts must be a list of 96 context names for plotting.")
            
    fig, ax = plt.subplots(figsize=(22, 11)) # Increased figsize slightly

    # Prepare bar colors: 16 bars for each of the 6 mutation types
    bar_colors_list = []
    for mut_type in mutation_types_order: # e.g., C>A, C>G, ...
        bar_colors_list.extend([colors_map.get(mut_type, '#000000')] * 16)
    
    # Ensure signature_series is aligned with ordered_contexts for plotting
    # This step is crucial if signature_series might not be perfectly ordered initially.
    # df_profiles was already reordered in main(), so signature_series should be ordered.
    # However, an explicit reindex here can be a safeguard.
    try:
        data_to_plot = signature_series.loc[ordered_contexts]
    except KeyError:
        print(f"Error for {signature_name}: Some contexts in ordered_contexts not found in signature_series. Plotting might be incorrect.")
        # Fallback to using the series as-is, assuming it's already ordered and complete
        if len(signature_series) == 96:
            data_to_plot = signature_series
        else:
            raise ValueError(f"Signature series for {signature_name} does not have 96 values and key errors occurred.")


    ax.bar(ordered_contexts, data_to_plot, color=bar_colors_list, width=0.75) # Adjusted width

    # Set X-axis Labels and Ticks (simplified: A_C>A_G style)
    simplified_labels = [f"{ctx[0]}_{ctx[6]}" for ctx in ordered_contexts] # A_G for A[C>A]G
    
    ax.set_xticks(range(len(ordered_contexts)))
    ax.set_xticklabels(simplified_labels, rotation=90, ha="center", fontsize=6, fontfamily='monospace')
    ax.tick_params(axis='x', which='major', pad=2) # Adjust padding for x-labels

    # Set Title and Y-label
    ax.set_title(signature_name, fontsize=18, pad=35) # Increased pad for title
    ax.set_ylabel("Contribution", fontsize=16) # Changed from Probability to Contribution
    ax.set_xlabel("Trinucleotide Context", fontsize=16, labelpad=10)


    # Add Visual Group Separators and Decorations
    # Vertical lines between the 6 main mutation groups
    for i in range(1, 6): # 5 lines for 6 groups
        ax.axvline(x=16*i - 0.5, color='grey', linestyle='-', linewidth=0.6)

    # Colored text headers for each of the 6 mutation groups
    y_pos_text = ax.get_ylim()[1] 
    if y_pos_text == 0 : # Handle case where all values are zero
        y_pos_text = 0.01 
        ax.set_ylim(0, y_pos_text) # Adjust ylim if all zero to make space for text

    for j, mut_type in enumerate(mutation_types_order):
        ax.text(
            j*16 + (16/2 - 0.5),  # Center text in the middle of the 16 bars
            y_pos_text * 1.02,    # Position text slightly above max y-value
            mut_type,
            ha='center',
            va='bottom',
            fontsize=14,
            color=colors_map.get(mut_type, '#000000'),
            fontweight='bold'
        )
    
    # General plot aesthetics
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linestyle='--', linewidth=0.5, color='grey', alpha=0.7)
    ax.set_xlim(-0.75, len(ordered_contexts)-0.25) # Adjust xlim to prevent bars cutoff

    # Layout and Save
    plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust rect to make space for main title if needed
    try:
        plt.savefig(output_path, dpi=300)
        # print(f"    Plot saved to {output_path}") # Already printed in main
    except Exception as e:
        print(f"    Error saving plot to {output_path}: {e}")
    finally:
        plt.close(fig) # Important to free memory


if __name__ == "__main__":
    main()
