"""
Calculates Tumor Mutational Burden (TMB) from MAF files for specified cohorts.

This script iterates through MAF files in a given directory structure,
counts Single Nucleotide Variants (SNVs) per sample for the specified cohorts,
and then normalizes these counts by a provided exome size to calculate TMB.
If exome_size_mb is 0, raw SNV counts are output.
"""
import pandas as pd
import argparse
import os
import glob
import sys
import gzip # For catching gzip.BadGzipFile

def create_parser():
    """Creates and returns the ArgumentParser object for the script."""
    parser = argparse.ArgumentParser(description="Calculate Tumor Mutational Burden (TMB) from MAF files.")
    parser.add_argument(
        "--maf_input_dir",
        type=str,
        required=True,
        help="Path to the root directory containing MAF files, typically organized by cohort subdirectories."
    )
    parser.add_argument(
        "--cohort_list",
        type=str,
        required=True,
        help="Comma-separated string of target cohort IDs (e.g., 'TCGA-LUAD,TCGA-SKCM')."
    )
    parser.add_argument(
        "--exome_size_mb",
        type=float,
        default=30.0,
        help="Estimated exome size in Megabases for TMB normalization. "
             "Set to 0 to output raw SNV counts instead of TMB. Default: 30.0 MB."
    )
    parser.add_argument(
        "--output_tmb_file",
        type=str,
        required=True,
        help="Path to save the output TMB results TSV file (e.g., './results/tables/tmb_by_sample.tsv')."
    )
    return parser

def is_valid_snv(row, ref_col='Reference_Allele', alt_col='Tumor_Seq_Allele2'):
    """
    Checks if a variant row represents a valid SNV.
    - Reference_Allele and Tumor_Seq_Allele2 must be single characters from A, C, G, T.
    - Reference_Allele must not be equal to Tumor_Seq_Allele2.
    """
    ref_allele = str(row[ref_col])
    alt_allele = str(row[alt_col])
    valid_bases = {'A', 'C', 'G', 'T'}
    
    is_snv_type = len(ref_allele) == 1 and ref_allele in valid_bases and \
                  len(alt_allele) == 1 and alt_allele in valid_bases and \
                  ref_allele != alt_allele
    return is_snv_type


def process_maf_file_for_snvs(maf_file_path, sample_snv_counts):
    """
    Reads a single MAF file, validates it, and counts SNVs for each sample.
    Updates the sample_snv_counts dictionary in place.
    """
    print(f"  Processing MAF file: {maf_file_path}")
    try:
        compression = 'gzip' if maf_file_path.endswith('.gz') else None
        maf_df = pd.read_csv(
            maf_file_path, sep='\t', comment='#', low_memory=False, compression=compression
        )
    except FileNotFoundError:
        print(f"    Error: MAF file not found: {maf_file_path}", file=sys.stderr); return
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as e:
        print(f"    Error parsing MAF file {maf_file_path}: {e}", file=sys.stderr); return
    except (gzip.BadGzipFile, EOFError) as e:
        print(f"    Error: MAF file {maf_file_path} is corrupted: {e}", file=sys.stderr); return
    except Exception as e:
        print(f"    Unexpected error reading MAF {maf_file_path}: {e}", file=sys.stderr); return

    required_cols = ['Tumor_Sample_Barcode', 'Variant_Type', 'Reference_Allele', 'Tumor_Seq_Allele2']
    missing_cols = [col for col in required_cols if col not in maf_df.columns]
    if missing_cols:
        print(f"    Error: MAF {maf_file_path} missing columns: {', '.join(missing_cols)}. Skipping.", file=sys.stderr)
        return

    snp_df = maf_df[maf_df['Variant_Type'] == 'SNP']
    if snp_df.empty: return

    ref_is_valid_base = snp_df['Reference_Allele'].astype(str).str.match(r'^[ACGT]$')
    alt_is_valid_base = snp_df['Tumor_Seq_Allele2'].astype(str).str.match(r'^[ACGT]$')
    ref_ne_alt = snp_df['Reference_Allele'] != snp_df['Tumor_Seq_Allele2']
    true_snvs_df = snp_df[ref_is_valid_base & alt_is_valid_base & ref_ne_alt]

    if true_snvs_df.empty: return
    
    snv_counts_in_maf = true_snvs_df['Tumor_Sample_Barcode'].value_counts()
    for sample_id, count in snv_counts_in_maf.items():
        sample_snv_counts[sample_id] = sample_snv_counts.get(sample_id, 0) + count

def _run_tmb_pipeline(args, processed_cohort_ids):
    """
    Core TMB calculation pipeline after args and cohorts are processed.
    This function contains the MAF processing, TMB calculation, and output saving.
    """
    sample_snv_counts = {}
    print(f"\nInitialized empty sample_snv_counts dictionary.")

    print("\nProcessing MAF files for SNV counts...")
    for cohort_id in processed_cohort_ids:
        print(f"Processing cohort: {cohort_id}")
        cohort_maf_dir = os.path.join(args.maf_input_dir, cohort_id)

        if not os.path.isdir(cohort_maf_dir):
            print(f"  Warning: MAF directory not found for cohort {cohort_id} at {cohort_maf_dir}. Skipping cohort.", file=sys.stderr)
            continue

        maf_files_gz = glob.glob(os.path.join(cohort_maf_dir, '*.maf.gz'))
        maf_files_plain = glob.glob(os.path.join(cohort_maf_dir, '*.maf'))
        all_maf_files = maf_files_gz + maf_files_plain

        if not all_maf_files:
            print(f"  Warning: No MAF files (.maf or .maf.gz) found for cohort {cohort_id} in {cohort_maf_dir}. Skipping cohort.", file=sys.stderr)
            continue
        
        print(f"  Found {len(all_maf_files)} MAF file(s) for cohort {cohort_id}.")
        for maf_file_path in all_maf_files:
            process_maf_file_for_snvs(maf_file_path, sample_snv_counts)
            
    print("\nFinished processing all MAF files.")

    num_processed_samples = len(sample_snv_counts)
    print(f"Processed {num_processed_samples} samples in total.")

    if not sample_snv_counts:
        print("No samples found or no SNVs counted. Output file will not be created.")
        print("\nScript finished.")
        return 

    print("\nCreating DataFrame from SNV counts...")
    try:
        tmb_df = pd.DataFrame(list(sample_snv_counts.items()), columns=['Tumor_Sample_Barcode', 'Total_SNVs'])
        tmb_df = tmb_df.sort_values(by='Tumor_Sample_Barcode').reset_index(drop=True)
        print(f"DataFrame created with {len(tmb_df)} samples.")
    except Exception as e:
        print(f"Error creating DataFrame from sample_snv_counts: {e}", file=sys.stderr)
        sys.exit(1) # Exit here as this is a critical data processing step

    if args.exome_size_mb > 0:
        print(f"Calculating TMB per Mb using exome size: {args.exome_size_mb} Mb.")
        tmb_df['TMB_mut_per_Mb'] = tmb_df['Total_SNVs'] / args.exome_size_mb
    else:
        print("Exome size is 0 or not provided. Outputting raw SNV counts.")

    print(f"\nPreparing to save TMB data to: {args.output_tmb_file}")
    output_dir = os.path.dirname(args.output_tmb_file)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
            print(f"Created output directory for TMB file: {output_dir}")
        except Exception as e:
            print(f"Error creating output directory {output_dir} for TMB file: {e}", file=sys.stderr)
            sys.exit(1) # Exit as we cannot save the output
    
    try:
        tmb_df.to_csv(args.output_tmb_file, sep='\t', index=False)
        print(f"TMB data saved to: {args.output_tmb_file}")
    except IOError as e:
        print(f"Error saving TMB data to {args.output_tmb_file}: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e: 
        print(f"An unexpected error occurred while saving TMB data: {e}", file=sys.stderr)
        sys.exit(1)
    
    print("\nScript finished successfully.")


def main(args_list=None):
    """
    Main function to parse arguments, process MAF files, calculate TMB, and save results.
    """
    parser = create_parser()
    try:
        args = parser.parse_args(args_list if args_list is not None else sys.argv[1:])
    except SystemExit as e: 
        # Argparse by default prints help and exits on error.
        # This catch is mostly for testing or if exit behavior needs customization.
        # print(f"Argument parsing error: {e}", file=sys.stderr) # Argparse handles its own error messages
        sys.exit(2) # Standard exit code for command-line syntax errors

    print("Parsed arguments:")
    print(f"  MAF Input Directory: {args.maf_input_dir}")
    print(f"  Cohort List String: {args.cohort_list}")
    print(f"  Exome Size (MB): {args.exome_size_mb}")
    print(f"  Output TMB File: {args.output_tmb_file}")

    processed_cohort_ids = [cohort.strip() for cohort in args.cohort_list.split(',') if cohort.strip()]
    if not processed_cohort_ids:
        print("Error: Cohort list is empty or invalid. Please provide a comma-separated list of cohort IDs.", file=sys.stderr)
        sys.exit(1) # Exit if no valid cohorts are provided
    print(f"  Processed Target Cohorts: {processed_cohort_ids}")

    # Call the core pipeline function
    _run_tmb_pipeline(args, processed_cohort_ids)


if __name__ == "__main__":
    main()
