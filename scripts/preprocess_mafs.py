import pandas as pd
import argparse
import os
import glob
from pyfaidx import Fasta, FastaIndexingError

# Define the complement map globally or pass it around
COMPLEMENT_MAP = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'} # Handle N for completeness

def get_trinucleotide_context(row, ref_genome):
    """
    Generates the trinucleotide context and pyrimidine-normalized mutation type for an SNV.
    """
    chrom = str(row['Chromosome'])
    start_pos = int(row['Start_Position']) # MAF is 1-based
    ref_allele_maf = row['Reference_Allele']
    alt_allele_maf = row['Tumor_Seq_Allele2']
    
    # 1. Chromosome Name Normalization
    if not chrom.startswith('chr'):
        normalized_chrom = 'chr' + chrom
    else:
        normalized_chrom = chrom
    
    # Validate chromosome existence in FASTA
    if normalized_chrom not in ref_genome:
        # Attempt to remove 'chr' if original didn't have it and normalized version not found
        if chrom.startswith('chr') and chrom[3:] in ref_genome:
            normalized_chrom = chrom[3:]
        elif not chrom.startswith('chr') and chrom in ref_genome: # Original name was fine
             normalized_chrom = chrom
        else:
            return pd.Series([None, None, f"Chromosome {normalized_chrom} not in FASTA"])


    # 2. Extract Trinucleotide Sequence (MAF Start_Position is 1-based)
    # pyfaidx uses 0-based indexing for slicing.
    # Upstream base: start_pos - 2 in 0-based index (i.e., (start_pos - 1) - 1)
    # Ref base: start_pos - 1 in 0-based index
    # Downstream base: start_pos in 0-based index
    trinuc_start_0based = start_pos - 2
    trinuc_end_0based = start_pos + 1 # Slicing is exclusive at the end

    try:
        if trinuc_start_0based < 0: # Check for start of chromosome
             return pd.Series([None, None, "Position too close to chromosome start"])
        
        trinucleotide_sequence = ref_genome[normalized_chrom][trinuc_start_0based:trinuc_end_0based].seq.upper()
        
        if len(trinucleotide_sequence) != 3:
            return pd.Series([None, None, f"Extracted sequence length not 3: {trinucleotide_sequence}"])

        ref_allele_fasta = trinucleotide_sequence[1] # Middle base is the reference
        
        # Verify MAF ref allele matches FASTA ref allele
        if ref_allele_maf != ref_allele_fasta:
            return pd.Series([None, None, f"FASTA/MAF ref mismatch: {ref_allele_fasta} vs {ref_allele_maf}"])

    except FastaIndexingError as e:
        return pd.Series([None, None, f"FastaIndexingError: {e}"])
    except KeyError: # If normalized_chrom somehow still isn't in ref_genome after checks
        return pd.Series([None, None, f"Chromosome {normalized_chrom} not in FASTA (extraction step)"])
    except Exception as e: # Catch any other unexpected errors during sequence extraction
        return pd.Series([None, None, f"Unexpected error in extraction: {e}"])

    # 3. Determine Mutation Type
    mutation_type_orig = f"{ref_allele_maf}>{alt_allele_maf}"

    # 4. Normalize to Pyrimidine Context
    ref_pyrimidine = ref_allele_maf
    alt_pyrimidine = alt_allele_maf
    context_pyrimidine = trinucleotide_sequence

    if ref_allele_maf in ['G', 'A']: # If purine, convert to pyrimidine context
        ref_pyrimidine = COMPLEMENT_MAP.get(ref_allele_maf, 'N')
        alt_pyrimidine = COMPLEMENT_MAP.get(alt_allele_maf, 'N')
        # Reverse and complement the trinucleotide sequence
        context_pyrimidine = "".join([COMPLEMENT_MAP.get(base, 'N') for base in trinucleotide_sequence[::-1]])
    
    if 'N' in [ref_pyrimidine, alt_pyrimidine] or 'N' in context_pyrimidine:
        return pd.Series([None, None, "N base in context/alleles"])


    # 5. Construct 96-Context String
    # Format: Upstream[Ref_Pyrimidine>Var_Pyrimidine]Downstream
    upstream_base = context_pyrimidine[0]
    downstream_base = context_pyrimidine[2]
    
    # The reference in the middle of the context string should always be the pyrimidine one
    # The variant is relative to this pyrimidine reference
    mutation_type_norm = f"{ref_pyrimidine}>{alt_pyrimidine}"
    trinuc_context_96 = f"{upstream_base}[{mutation_type_norm}]{downstream_base}"
    
    return pd.Series([mutation_type_norm, trinuc_context_96, None])


def main():
    parser = argparse.ArgumentParser(description="Preprocess MAF files to generate a mutation catalog.")
    parser.add_argument("--maf_input_dir", required=True,
                        help="Path to the base directory containing cohort subdirectories with MAF files.")
    parser.add_argument("--ref_genome_fasta", required=True,
                        help="Path to the reference genome FASTA file (e.g., hg19.fa).")
    parser.add_argument("--output_matrix_file", required=True,
                        help="Path for the output CSV mutation catalog file.")

    args = parser.parse_args()

    print(f"MAF input directory: {args.maf_input_dir}")
    print(f"Reference genome FASTA: {args.ref_genome_fasta}")
    print(f"Output matrix file: {args.output_matrix_file}")

    # Initialize Fasta object
    try:
        print(f"Loading reference genome from: {args.ref_genome_fasta}")
        # sequence_always_upper=True is default in more recent pyfaidx, but good to be explicit
        ref_genome = Fasta(args.ref_genome_fasta, sequence_always_upper=True)
        print("Reference genome loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Reference genome FASTA file not found: {args.ref_genome_fasta}")
        return
    except FastaIndexingError as e: # Catch errors like empty or malformed FASTA
        print(f"Error: Could not index or read reference FASTA: {args.ref_genome_fasta}. Error: {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred while loading the reference FASTA: {e}")
        return

    # Identify cohort subdirectories
    try:
        cohort_dirs = [d for d in os.listdir(args.maf_input_dir)
                       if os.path.isdir(os.path.join(args.maf_input_dir, d))]
    except FileNotFoundError:
        print(f"Error: MAF input directory not found: {args.maf_input_dir}")
        return
    except Exception as e:
        print(f"Error accessing MAF input directory {args.maf_input_dir}: {e}")
        return

    if not cohort_dirs:
        print(f"No cohort subdirectories found in {args.maf_input_dir}")
        return

    print(f"Found cohort directories: {', '.join(cohort_dirs)}")

    all_maf_data = [] # To store DataFrames from each MAF file
    processed_files_count = 0
    skipped_snvs_context = 0

    for cohort_id in cohort_dirs:
        cohort_path = os.path.join(args.maf_input_dir, cohort_id)
        print(f"\nProcessing cohort directory: {cohort_path}")

        # Find MAF files in the cohort directory
        # Flexible patterns: .maf, .maf.gz, .maf.txt, .maf.txt.gz etc.
        maf_file_patterns = [
            os.path.join(cohort_path, '*.maf'),
            os.path.join(cohort_path, '*.maf.gz'),
            os.path.join(cohort_path, '*.maf.txt'),
            os.path.join(cohort_path, '*.maf.txt.gz'),
            os.path.join(cohort_path, '*.maf.*.gz') # More generic for some TCGA archives
        ]
        
        maf_files_found = []
        for pattern in maf_file_patterns:
            maf_files_found.extend(glob.glob(pattern))
        
        # Remove duplicates if patterns overlap
        maf_files_found = sorted(list(set(maf_files_found)))

        if not maf_files_found:
            print(f"No MAF files found in {cohort_path} with patterns: {maf_file_patterns}")
            continue

        for maf_file_path in maf_files_found:
            print(f"  Processing MAF file: {maf_file_path}")
            try:
                # Determine compression based on file extension
                compression = 'gzip' if maf_file_path.endswith('.gz') else None
                maf_df = pd.read_csv(maf_file_path, sep='\t', comment='#', low_memory=False,
                                     compression=compression,
                                     # Define dtypes for columns that might be problematic or for consistency
                                     # For example, Chromosome can be 'X', 'Y', 'MT'
                                     dtype={'Chromosome': str, 'Start_Position': pd.Int64Dtype(), 
                                            'End_Position': pd.Int64Dtype()})
                print(f"    Read {len(maf_df)} rows from {maf_file_path}.")

                # Robustly identify essential columns
                # Define primary and potential alternative names
                column_map_options = {
                    'Hugo_Symbol': ['Hugo_Symbol'],
                    'Chromosome': ['Chromosome', 'chr', 'Chr'],
                    'Start_Position': ['Start_Position', 'Start_position', 'start'],
                    # 'End_Position': ['End_Position', 'End_position', 'end'], # Not strictly needed for SNV context
                    'Reference_Allele': ['Reference_Allele', 'Ref_allele'],
                    'Tumor_Seq_Allele1': ['Tumor_Seq_Allele1', 't_ref_count'], # MAF may use this as ref
                    'Tumor_Seq_Allele2': ['Tumor_Seq_Allele2', 't_alt_count', 'Tumor_Allele'], # MAF usually has this as alt
                    'Variant_Type': ['Variant_Type', 'Mutation_Type'],
                    # 'Variant_Classification': ['Variant_Classification'],
                    'Tumor_Sample_Barcode': ['Tumor_Sample_Barcode', 'Sample_ID', 'sample_id']
                }
                
                current_columns = {}
                missing_critical_columns = False
                essential_cols_for_snv = ['Chromosome', 'Start_Position', 'Reference_Allele', 
                                          'Tumor_Seq_Allele2', 'Variant_Type', 'Tumor_Sample_Barcode']

                for essential_col, options in column_map_options.items():
                    found_col = None
                    for option in options:
                        if option in maf_df.columns:
                            current_columns[essential_col] = option
                            found_col = option
                            break
                    if not found_col and essential_col in essential_cols_for_snv:
                        print(f"    Warning: Critical column '{essential_col}' (or alternatives like {options}) not found in {maf_file_path}. Skipping file.")
                        missing_critical_columns = True
                        break 
                    elif not found_col:
                         print(f"    Info: Optional column '{essential_col}' (or alternatives like {options}) not found in {maf_file_path}.")


                if missing_critical_columns:
                    continue

                # Filter for Somatic Single Nucleotide Variants (SNVs)
                # 1. Variant_Type is 'SNP'
                snv_df = maf_df[maf_df[current_columns['Variant_Type']] == 'SNP'].copy() # Use .copy() to avoid SettingWithCopyWarning

                # 2. Reference_Allele and Tumor_Seq_Allele2 are single characters [A, C, G, T]
                valid_bases = ['A', 'C', 'G', 'T']
                snv_df = snv_df[
                    snv_df[current_columns['Reference_Allele']].isin(valid_bases) &
                    snv_df[current_columns['Tumor_Seq_Allele2']].isin(valid_bases)
                ]

                # 3. Reference_Allele and Tumor_Seq_Allele2 are not the same
                snv_df = snv_df[
                    snv_df[current_columns['Reference_Allele']] != snv_df[current_columns['Tumor_Seq_Allele2']]
                ]
                
                num_snvs = len(snv_df)
                print(f"    Found {num_snvs} SNVs after filtering in {maf_file_path}.")

                if num_snvs > 0:
                    # Select and rename columns to a standard set for easier concatenation later
                    # This should use the `current_columns` mapping to select the correct original column names
                    processed_snv_df = pd.DataFrame({
                        'Hugo_Symbol': snv_df[current_columns['Hugo_Symbol']] if 'Hugo_Symbol' in current_columns else 'Unknown',
                        'Chromosome': snv_df[current_columns['Chromosome']],
                        'Start_Position': snv_df[current_columns['Start_Position']],
                        'Reference_Allele': snv_df[current_columns['Reference_Allele']],
                        'Tumor_Seq_Allele2': snv_df[current_columns['Tumor_Seq_Allele2']],
                        'Variant_Type': snv_df[current_columns['Variant_Type']],
                        'Tumor_Sample_Barcode': snv_df[current_columns['Tumor_Sample_Barcode']],
                        'Cohort': cohort_id # Add cohort information
                    })
                    all_maf_data.append(processed_snv_df)
                    processed_files_count += 1 # Increment for files contributing SNVs

            except (IOError, pd.errors.EmptyDataError, pd.errors.ParserError) as e:
                print(f"    Error reading or parsing MAF file {maf_file_path}: {e}. Skipping file.")
                continue
            except KeyError as e:
                print(f"    KeyError (likely missing a critical column that wasn't caught by initial check) in {maf_file_path}: {e}. Skipping file.")
                continue
            except Exception as e: # Catch any other unexpected errors
                print(f"    An unexpected error occurred while processing MAF file {maf_file_path}: {e}. Skipping file.")
                continue
    if not all_maf_data:
        print("\nNo valid SNV data collected from any MAF files.")
        if processed_files_count == 0:
            print("No MAF files were successfully processed.")
        return # Exit if no data

    print(f"\nConcatenating data from {len(all_maf_data)} processed SNV sets...")
    final_matrix = pd.concat(all_maf_data, ignore_index=True)
    print(f"Total SNVs collected before context generation: {len(final_matrix)}")

    if final_matrix.empty:
        print("No SNVs available for trinucleotide context generation.")
        return

    print("\nGenerating trinucleotide contexts...")
    # Apply the function to each row
    # The result of apply will be a DataFrame with columns: 0 (mutation_type), 1 (trinucleotide_context), 2 (error_reason)
    context_results = final_matrix.apply(
        lambda row: get_trinucleotide_context(row, ref_genome), axis=1
    )
    final_matrix[['mutation_type', 'trinucleotide_context', 'context_error']] = context_results
    
    # Log errors during context generation
    context_errors = final_matrix[final_matrix['context_error'].notna()]
    if not context_errors.empty:
        print(f"\nEncountered {len(context_errors)} errors during trinucleotide context generation:")
        # Log a sample of errors
        for index, row in context_errors.head().iterrows():
            print(f"  SNV at {row['Cohort']}:{row['Chromosome']}:{row['Start_Position']} - Error: {row['context_error']}")
        # For more detailed per-file logging, this would need to be inside the file loop or errors aggregated differently
    
    skipped_snvs_context = len(context_errors)
    final_matrix.dropna(subset=['trinucleotide_context'], inplace=True) # Remove rows where context could not be generated
    
    print(f"Successfully generated trinucleotide contexts for {len(final_matrix)} SNVs.")
    print(f"Skipped {skipped_snvs_context} SNVs due to issues in context generation (e.g., FASTA mismatch, edge of chrom).")

    if final_matrix.empty:
        print("No SNVs remaining after trinucleotide context generation and error filtering.")
        return

    # Display a sample of generated contexts if successful
    if not final_matrix.empty:
        print("\nSample of generated contexts (first 5):")
        print(final_matrix[['Tumor_Sample_Barcode', 'Chromosome', 'Start_Position', 'Reference_Allele', 'Tumor_Seq_Allele2', 'mutation_type', 'trinucleotide_context']].head())
    
    print(f"\nTotal SNVs with valid trinucleotide context: {len(final_matrix)}")

    # 1. Define the 96 Standard Contexts
    bases = ['A', 'C', 'G', 'T']
    substitution_types = { # pyrimidine_ref -> list of alts
        'C': ['A', 'G', 'T'],
        'T': ['A', 'C', 'G']
    }
    standard_96_contexts = []
    for ref_pyr, alt_bases in substitution_types.items():
        for alt_pyr in alt_bases:
            for upstream in bases:
                for downstream in bases:
                    standard_96_contexts.append(f"{upstream}[{ref_pyr}>{alt_pyr}]{downstream}")
    standard_96_contexts.sort() # Ensure lexicographical order
    print(f"DEBUG: Corrected number of generated standard contexts: {len(standard_96_contexts)}") # Should be 96

    print(f"DEBUG: Number of generated standard contexts: {len(standard_96_contexts)}")
    print(f"DEBUG: First 5 contexts: {standard_96_contexts[:5]}")

    # 2. Aggregate Mutation Counts
    if not final_matrix.empty:
        print("\nAggregating mutation counts per sample...")
        mutation_counts = final_matrix.groupby(
            ['Tumor_Sample_Barcode', 'trinucleotide_context']
        ).size().unstack(fill_value=0)
        
        print(f"Found {mutation_counts.shape[0]} unique samples with mutations.")

        # 3. Reindex to Standard Contexts
        print(f"Reindexing columns to {len(standard_96_contexts)} standard contexts...")
        mutation_matrix = mutation_counts.reindex(columns=standard_96_contexts, fill_value=0)
        
        # Ensure Tumor_Sample_Barcode is the index name
        mutation_matrix.index.name = 'Tumor_Sample_Barcode'

        print(f"Final mutation matrix dimensions: {mutation_matrix.shape}")
        print(f"Columns in final matrix: {mutation_matrix.columns.tolist()[:5]}... (first 5 shown)")

        # 4. Logging (additional) - e.g. samples with no mutations in the 96 contexts
        samples_with_no_muts_in_96_contexts = mutation_matrix[mutation_matrix.sum(axis=1) == 0]
        if not samples_with_no_muts_in_96_contexts.empty:
            print(f"Warning: {len(samples_with_no_muts_in_96_contexts)} samples have zero mutations across the 96 standard contexts.")
            # print(samples_with_no_muts_in_96_contexts.index.tolist())

        output_file_path = args.output_matrix_file
        output_dir_for_matrix = os.path.dirname(output_file_path)
        
        if output_dir_for_matrix and not os.path.exists(output_dir_for_matrix):
            print(f"Creating output directory for matrix: {output_dir_for_matrix}")
            os.makedirs(output_dir_for_matrix, exist_ok=True)

        print(f"\nSaving final mutation catalog to: {output_file_path}")
        try:
            mutation_matrix.to_csv(output_file_path) # Use output_file_path here
            print(f"Successfully saved mutation matrix to {output_file_path}")
        except IOError as e:
            print(f"Error: Could not save mutation matrix to {args.output_matrix_file}. Error: {e}")
                
    else:
        print("\nNo data to aggregate. Skipping matrix generation and saving.")

    print("\nScript finished.")


if __name__ == "__main__":
    main()
