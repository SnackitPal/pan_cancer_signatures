import pandas as pd
import argparse
import os
import glob

def main():
    parser = argparse.ArgumentParser(description="Generate a mapping of Tumor_Sample_Barcode to Cohort_ID from MAF files.")
    parser.add_argument("--maf_input_dir", type=str, required=True,
                        help="Path to the base directory containing cohort-specific MAF subdirectories.")
    parser.add_argument("--output_map_file", type=str, required=True,
                        help="Path for the output CSV mapping file (Tumor_Sample_Barcode, Cohort_ID).")
    
    args = parser.parse_args()

    # Print input parameters
    print("Starting sample-cohort map generation script with the following parameters:")
    print(f"  MAF input directory: {args.maf_input_dir}")
    print(f"  Output map file: {args.output_map_file}")

    # Initialize list to store pairs
    sample_cohort_pairs = []

    # Check if MAF input directory exists
    if not os.path.exists(args.maf_input_dir):
        print(f"Error: MAF input directory not found: {args.maf_input_dir}")
        return # Exit script
    if not os.path.isdir(args.maf_input_dir):
        print(f"Error: MAF input path is not a directory: {args.maf_input_dir}")
        return # Exit script

    print(f"\nScanning MAF input directory: {args.maf_input_dir}")

    try:
        entries = os.listdir(args.maf_input_dir)
    except OSError as e:
        print(f"Error listing contents of MAF input directory {args.maf_input_dir}: {e}")
        return

    for entry_name in entries:
        cohort_dir_path = os.path.join(args.maf_input_dir, entry_name)
        
        if os.path.isdir(cohort_dir_path):
            cohort_id = entry_name # Use the directory name as Cohort_ID
            print(f"  Processing cohort directory: {cohort_id} (Path: {cohort_dir_path})")
            
            # Define potential MAF file patterns
            maf_file_patterns = [
                os.path.join(cohort_dir_path, '*.maf.gz'),
                os.path.join(cohort_dir_path, '*.maf'),
                os.path.join(cohort_dir_path, '*.maf.txt.gz'),
                os.path.join(cohort_dir_path, '*.maf.txt')
            ]
            
            maf_files_found = []
            for pattern in maf_file_patterns:
                maf_files_found.extend(glob.glob(pattern))
            
            maf_files_found = sorted(list(set(maf_files_found))) # Unique sorted list

            if not maf_files_found:
                print(f"    No MAF files found in {cohort_dir_path} with patterns: {maf_file_patterns}")
                continue

            cohort_barcodes = set()
            potential_barcode_cols = ['Tumor_Sample_Barcode', 'Sample_ID', 'sample_id']

            for maf_file_path in maf_files_found:
                print(f"    Processing MAF file: {os.path.basename(maf_file_path)}")
                actual_barcode_col_name = None
                try:
                    # 1. Read header to find barcode column
                    header_df = pd.read_csv(maf_file_path, sep='\t', comment='#', nrows=0, compression='gzip' if maf_file_path.endswith('.gz') else None)
                    
                    for col_name in potential_barcode_cols:
                        if col_name in header_df.columns:
                            actual_barcode_col_name = col_name
                            break
                    
                    if not actual_barcode_col_name:
                        print(f"      Warning: Could not find a suitable barcode column (e.g., Tumor_Sample_Barcode) in {maf_file_path}. Skipping file.")
                        continue

                    # 2. Read only the identified barcode column
                    df_maf = pd.read_csv(
                        maf_file_path, 
                        sep='\t', 
                        comment='#', 
                        usecols=[actual_barcode_col_name], 
                        low_memory=False, # Good practice for MAFs, though usecols helps
                        compression='gzip' if maf_file_path.endswith('.gz') else None
                    )
                    
                    # 3. Extract unique barcodes
                    if not df_maf.empty and actual_barcode_col_name in df_maf.columns:
                        unique_in_file = df_maf[actual_barcode_col_name].unique()
                        cohort_barcodes.update(unique_in_file)
                    else:
                        print(f"      Warning: No data or barcode column not found after reading {maf_file_path}.")

                except pd.errors.EmptyDataError:
                    print(f"      Warning: MAF file {maf_file_path} is empty. Skipping.")
                    continue
                except Exception as e:
                    print(f"      Error processing MAF file {maf_file_path}: {e}. Skipping file.")
                    continue
            
            # Add pairs for this cohort
            for barcode in cohort_barcodes:
                sample_cohort_pairs.append((barcode, cohort_id))
            
            print(f"    Found {len(cohort_barcodes)} unique Tumor_Sample_Barcodes in cohort {cohort_id}")

    # After iterating through all cohort directories
    if not sample_cohort_pairs:
        print("\nNo Tumor_Sample_Barcodes found in any MAF files. Output file will not be created.")
        return

    print("\nCreating and saving the sample-cohort map...")
    df_map = pd.DataFrame(sample_cohort_pairs, columns=['Tumor_Sample_Barcode', 'Cohort'])

    # Handle Duplicates
    total_pairs = len(df_map)
    df_map.drop_duplicates(subset=['Tumor_Sample_Barcode'], keep='first', inplace=True)
    unique_barcodes_count = len(df_map)

    if total_pairs > unique_barcodes_count:
        print(f"Warning: Found and removed {total_pairs - unique_barcodes_count} duplicate Tumor_Sample_Barcode entries. "
              "Each barcode is now mapped to its first encountered cohort.")

    # Save Output
    try:
        output_dir = os.path.dirname(args.output_map_file)
        if output_dir and not os.path.exists(output_dir): # Check if output_dir is not empty string
            os.makedirs(output_dir, exist_ok=True)
            print(f"  Created output directory: {output_dir}")
        
        df_map.to_csv(args.output_map_file, index=False)
        print(f"  Total unique Tumor_Sample_Barcodes mapped: {unique_barcodes_count}")
        print(f"  Sample to Cohort map saved to: {args.output_map_file}")
    except IOError as e:
        print(f"Error: Could not save output map file to {args.output_map_file}. Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while saving the output map file: {e}")

    print("\nSample-cohort map generation finished.")


if __name__ == "__main__":
    main()
