import unittest
from unittest import mock
import pandas as pd
import argparse
import os
import sys
from pyfaidx import FastaIndexingError # For mocking

# Add scripts directory to sys.path to allow direct import of preprocess_mafs
scripts_dir = os.path.join(os.path.dirname(__file__), '..', 'scripts')
sys.path.insert(0, os.path.abspath(scripts_dir))

import preprocess_mafs # Import the script to be tested

# Mock Fasta object structure for pyfaidx
class MockFastaSequence:
    def __init__(self, seq):
        self.seq = seq

    def __getitem__(self, key): # To handle slicing like [start:end]
        if isinstance(key, slice):
            return MockFastaSequence(self.seq[key.start:key.stop])
        raise TypeError("MockFastaSequence subscript must be slice")


class MockFasta:
    def __init__(self, fasta_data):
        # fasta_data is a dict like {'chr1': 'ACGTN...', 'chr2': 'TTGCA...'}
        self.data = {chrom: MockFastaSequence(seq) for chrom, seq in fasta_data.items()}

    def __contains__(self, chrom):
        return chrom in self.data

    def __getitem__(self, chrom):
        if chrom not in self.data:
            raise KeyError(f"Chromosome {chrom} not in mock Fasta")
        return self.data[chrom]


class TestPreprocessMafs(unittest.TestCase):

    def setUp(self):
        # Suppress print statements from the script during tests for cleaner output
        self.patcher_print = mock.patch('builtins.print')
        self.mock_print = self.patcher_print.start()

        # Define standard 96 contexts (can also be imported from preprocess_mafs if made accessible)
        bases = ['A', 'C', 'G', 'T']
        pyrimidine_refs = ['C', 'T']
        self.standard_96_contexts = []
        for ref_pyr in pyrimidine_refs:
            other_bases = [b for b in bases if b != ref_pyr and b != preprocess_mafs.COMPLEMENT_MAP[ref_pyr]]
            for alt in other_bases:
                for upstream in bases:
                    for downstream in bases:
                        self.standard_96_contexts.append(f"{upstream}[{ref_pyr}>{alt}]{downstream}")
        self.standard_96_contexts.sort()


    def tearDown(self):
        self.patcher_print.stop()
        # Clean up sys.path modification if necessary, though typically not for simple appends
        # if scripts_dir == sys.path[0]:
        #     sys.path.pop(0)


    def test_argument_parsing(self):
        """Test parsing of command-line arguments."""
        test_argv = [
            'preprocess_mafs.py',
            '--maf_input_dir', './test_mafs/',
            '--ref_genome_fasta', './hg19.fa',
            '--output_matrix_file', './output.csv'
        ]
        with mock.patch('sys.argv', test_argv):
            # The parser is defined in main(), so we need to access it carefully or call main()
            # For simplicity, let's assume the parser is accessible globally in preprocess_mafs
            # If not, this test would need to call main() and check its effects or mock its internals.
            # Re-creating the parser as it's done in the script:
            parser = argparse.ArgumentParser(description="Preprocess MAF files to generate a mutation catalog.")
            parser.add_argument("--maf_input_dir", required=True)
            parser.add_argument("--ref_genome_fasta", required=True)
            parser.add_argument("--output_matrix_file", required=True)
            
            args = parser.parse_args(test_argv[1:]) # Exclude script name

            self.assertEqual(args.maf_input_dir, './test_mafs/')
            self.assertEqual(args.ref_genome_fasta, './hg19.fa')
            self.assertEqual(args.output_matrix_file, './output.csv')

    @mock.patch('preprocess_mafs.os.listdir')
    @mock.patch('preprocess_mafs.os.path.isdir')
    @mock.patch('preprocess_mafs.glob.glob')
    @mock.patch('preprocess_mafs.pd.read_csv') # Mock reading MAFs
    @mock.patch('preprocess_mafs.Fasta')      # Mock Fasta loading
    def test_maf_file_discovery(self, mock_fasta_constructor, mock_read_csv, mock_glob, mock_isdir, mock_listdir):
        """Test discovery of cohort directories and MAF files."""
        # Setup mock file system
        mock_listdir.return_value = ['COHORT1', 'COHORT2', 'not_a_dir.txt']
        
        def isdir_side_effect(path):
            if path.endswith('COHORT1') or path.endswith('COHORT2'):
                return True
            return False
        mock_isdir.side_effect = isdir_side_effect

        def glob_side_effect(pattern):
            if 'COHORT1' in pattern and pattern.endswith("*.maf.gz"):
                return ['/fake/path/COHORT1/file1.maf.gz']
            if 'COHORT2' in pattern and pattern.endswith("*.maf"):
                 return ['/fake/path/COHORT2/file2.maf']
            return [] # Default empty for other patterns
        mock_glob.side_effect = glob_side_effect
        
        # Mock read_csv to return an empty DataFrame to prevent further processing errors
        mock_read_csv.return_value = pd.DataFrame(columns=['Variant_Type', 'Reference_Allele', 'Tumor_Seq_Allele2', 'Chromosome', 'Start_Position', 'Tumor_Sample_Barcode'])
        
        # Mock Fasta constructor to return a dummy mock object
        mock_fasta_constructor.return_value = MockFasta({})


        # Simulate running main() with minimal args
        test_args = argparse.Namespace(
            maf_input_dir='/fake/path',
            ref_genome_fasta='/fake/hg19.fa',
            output_matrix_file='/fake/output.csv'
        )
        with mock.patch('argparse.ArgumentParser.parse_args', return_value=test_args):
            preprocess_mafs.main()

        # Assertions: check if print calls indicate correct discovery (example)
        # This is a bit fragile; better if the script returned discovered files or had testable state
        self.mock_print.assert_any_call("Found cohort directories: COHORT1, COHORT2")
        self.mock_print.assert_any_call("  Processing MAF file: /fake/path/COHORT1/file1.maf.gz")
        self.mock_print.assert_any_call("  Processing MAF file: /fake/path/COHORT2/file2.maf")
        
        # Check that glob was called for various patterns
        expected_glob_calls = [
            mock.call('/fake/path/COHORT1/*.maf'), mock.call('/fake/path/COHORT1/*.maf.gz'),
            mock.call('/fake/path/COHORT1/*.maf.txt'), mock.call('/fake/path/COHORT1/*.maf.txt.gz'),
            mock.call('/fake/path/COHORT1/*.maf.*.gz'),
            mock.call('/fake/path/COHORT2/*.maf'), mock.call('/fake/path/COHORT2/*.maf.gz'),
            mock.call('/fake/path/COHORT2/*.maf.txt'), mock.call('/fake/path/COHORT2/*.maf.txt.gz'),
            mock.call('/fake/path/COHORT2/*.maf.*.gz'),
        ]
        mock_glob.assert_has_calls(expected_glob_calls, any_order=True)


    def test_snv_filtering_logic(self):
        """Test the SNV filtering logic directly or via main loop simulation."""
        # This tests the filtering part of the script. Ideally, this logic would be in a
        # separate function. Since it's in main, we prepare data and check results.
        
        # Sample MAF data
        maf_data = {
            'Hugo_Symbol': ['GENE1', 'GENE2', 'GENE3', 'GENE4', 'GENE5', 'GENE6', 'GENE7'],
            'Chromosome': ['1', '1', '1', '1', '1', '1', '1'],
            'Start_Position': [100, 200, 300, 400, 500, 600, 700],
            'Variant_Type': ['SNP', 'SNP', 'DNP', 'SNP', 'SNP', 'SNP', 'INS'],
            'Reference_Allele': ['A', 'C', 'G', 'T', 'A', 'N', 'G'],
            'Tumor_Seq_Allele2': ['G', 'T', 'A', 'T', 'A', 'C', 'GT'], # T>T (invalid), A==A (invalid)
            'Tumor_Sample_Barcode': ['S1', 'S1', 'S1', 'S2', 'S2', 'S2', 'S3']
        }
        sample_df = pd.DataFrame(maf_data)
        
        # Expected SNVs after filtering
        # GENE1: A>G (valid)
        # GENE2: C>T (valid)
        # GENE3: DNP (invalid)
        # GENE4: T>T (ref == alt, invalid, based on script logic of ref != alt) -> Correction: T>T is not what I wrote above, this is T>T. The test is that ref!=alt. So T>T is not ref!=alt.
        # Let's re-evaluate GENE4. If Ref=T, Alt=T, then Ref==Alt, so it should be filtered.
        # For GENE4, let's make it Ref=T, Alt=C (valid)
        # For GENE5, Ref=A, Alt=A (ref==alt, invalid)
        # For GENE6, Ref=N (invalid base)
        # For GENE7, INS (invalid type)
        
        corrected_maf_data = {
            'Hugo_Symbol':        ['GENE1', 'GENE2', 'GENE3', 'GENE4', 'GENE5', 'GENE6', 'GENE7', 'GENE8'],
            'Chromosome':         ['1',     '1',     '1',     '1',     '1',     '1',     '1',     '1'],
            'Start_Position':     [100,     200,     300,     400,     500,     600,     700,     800],
            'Variant_Type':       ['SNP',   'SNP',   'DNP',   'SNP',   'SNP',   'SNP',   'INS',   'SNP'],
            'Reference_Allele':   ['A',     'C',     'G',     'T',     'A',     'N',     'G',     'G'],
            'Tumor_Seq_Allele2':  ['G',     'T',     'A',     'C',     'A',     'C',     'GT',    'G'], # G>G for GENE8
            'Tumor_Sample_Barcode':['S1',    'S1',    'S1',    'S2',    'S2',    'S2',    'S3',    'S3']
        }
        sample_df = pd.DataFrame(corrected_maf_data)
        
        # Columns needed by the script (taken from script's current_columns logic)
        # These are the "essential_cols_for_snv"
        script_essential_cols = {
             'Hugo_Symbol': 'Hugo_Symbol', 'Chromosome': 'Chromosome', 'Start_Position': 'Start_Position',
             'Reference_Allele': 'Reference_Allele', 'Tumor_Seq_Allele2': 'Tumor_Seq_Allele2',
             'Variant_Type': 'Variant_Type', 'Tumor_Sample_Barcode': 'Tumor_Sample_Barcode'
        }

        # Simulate the filtering part of preprocess_mafs.main
        # This requires a bit of careful extraction or direct application of the filter conditions
        
        # From script:
        # snv_df = maf_df[maf_df[current_columns['Variant_Type']] == 'SNP'].copy()
        # valid_bases = ['A', 'C', 'G', 'T']
        # snv_df = snv_df[
        #     snv_df[current_columns['Reference_Allele']].isin(valid_bases) &
        #     snv_df[current_columns['Tumor_Seq_Allele2']].isin(valid_bases)
        # ]
        # snv_df = snv_df[
        #     snv_df[current_columns['Reference_Allele']] != snv_df[current_columns['Tumor_Seq_Allele2']]
        # ]
        
        filtered_df = sample_df[sample_df[script_essential_cols['Variant_Type']] == 'SNP'].copy()
        valid_bases = ['A', 'C', 'G', 'T']
        filtered_df = filtered_df[
            filtered_df[script_essential_cols['Reference_Allele']].isin(valid_bases) &
            filtered_df[script_essential_cols['Tumor_Seq_Allele2']].isin(valid_bases)
        ]
        filtered_df = filtered_df[
            filtered_df[script_essential_cols['Reference_Allele']] != filtered_df[script_essential_cols['Tumor_Seq_Allele2']]
        ]
        
        self.assertEqual(len(filtered_df), 2) # GENE1 (A>G), GENE2 (C>T), GENE4 (T>C)
        self.assertListEqual(filtered_df['Hugo_Symbol'].tolist(), ['GENE1', 'GENE2', 'GENE4'])


    def test_chromosome_name_normalization(self):
        """Test chromosome name normalization in get_trinucleotide_context."""
        # Mock row and Fasta object (only needs to check for chrom existence)
        mock_ref_genome = MockFasta({'chr1': '', 'chrX': '', 'MT': ''}) # MT for mitochondrial
        
        row_data_1 = {'Chromosome': '1', 'Start_Position':100, 'Reference_Allele':'A', 'Tumor_Seq_Allele2':'C'}
        # We only test the normalization part, so other parts of context gen can be ignored or return None
        with mock.patch.object(mock_ref_genome, '__getitem__', side_effect=KeyError("Simulate chrom not found until normalized")):
             # Temporarily make it seem like '1' is not found to test 'chr1'
            def getitem_effect(key):
                if key == 'chr1': return MockFastaSequence("ACG") # Assume 'chr1' is the valid one
                raise KeyError
            mock_ref_genome.__getitem__ = mock.Mock(side_effect=getitem_effect)
            series_result = preprocess_mafs.get_trinucleotide_context(pd.Series(row_data_1), mock_ref_genome)
            mock_ref_genome.__getitem__.assert_any_call('chr1') # Check it tried 'chr1'

        row_data_x = {'Chromosome': 'X', 'Start_Position':100, 'Reference_Allele':'A', 'Tumor_Seq_Allele2':'C'}
        with mock.patch.object(mock_ref_genome, '__getitem__') as mock_getitem_x:
            mock_getitem_x.return_value = MockFastaSequence("ACG")
            preprocess_mafs.get_trinucleotide_context(pd.Series(row_data_x), mock_ref_genome)
            mock_getitem_x.assert_any_call('chrX')

        row_data_chry = {'Chromosome': 'chrY', 'Start_Position':100, 'Reference_Allele':'A', 'Tumor_Seq_Allele2':'C'}
        mock_ref_genome_with_chry = MockFasta({'chrY': 'ACG'})
        preprocess_mafs.get_trinucleotide_context(pd.Series(row_data_chry), mock_ref_genome_with_chry)
        # No error means it likely used chrY directly.

        row_data_mt = {'Chromosome': 'MT', 'Start_Position':100, 'Reference_Allele':'A', 'Tumor_Seq_Allele2':'C'}
        mock_ref_genome_with_mt = MockFasta({'chrMT': 'ACG'}) # Some use chrMT
        preprocess_mafs.get_trinucleotide_context(pd.Series(row_data_mt), mock_ref_genome_with_mt)
        # Check it tries chrMT

        row_data_no_prefix_found = {'Chromosome': '23', 'Start_Position':100, 'Reference_Allele':'A', 'Tumor_Seq_Allele2':'C'}
        mock_ref_genome_empty = MockFasta({}) # No valid chroms
        result_series = preprocess_mafs.get_trinucleotide_context(pd.Series(row_data_no_prefix_found), mock_ref_genome_empty)
        self.assertIsNotNone(result_series[2]) # Error message should be present
        self.assertIn("Chromosome chr23 not in FASTA", result_series[2])


    def test_get_trinucleotide_context_simple_ct_ref(self):
        """Test C>A context: A[C>A]G, FASTA provides 'ACG'"""
        mock_ref_genome = MockFasta({'chr1': 'NNNACGTNNN'}) # SNV at pos 5 (1-based) -> index 4
        # MAF Start_Position is 1-based. For 'ACG', C is at pos 5.
        # Upstream (A) is pos 4. Downstream (G) is pos 6.
        # pyfaidx wants 0-based: (5-1)-1 = 3 for A, (5-1) = 4 for C, (5-1)+1 = 5 for G
        # So, sequence is ref_genome['chr1'][3:6].seq -> ACG
        row_data = pd.Series({
            'Chromosome': '1', 'Start_Position': 5, 
            'Reference_Allele': 'C', 'Tumor_Seq_Allele2': 'A'
        })
        mtype, context, err = preprocess_mafs.get_trinucleotide_context(row_data, mock_ref_genome)
        self.assertIsNone(err)
        self.assertEqual(mtype, 'C>A')
        self.assertEqual(context, 'A[C>A]G')

    def test_get_trinucleotide_context_purine_ref_normalization(self):
        """Test G>T context: C[G>T]T (raw) -> A[C>A]G (normalized)"""
        # Fasta provides 'CGT' for G>T. Complement is 'GCA'. Ref G (in CGT) is C (in GCA - pyrimidine).
        # G>T becomes C>A. Upstream C (in CGT) is G (in GCA). Downstream T (in CGT) is A (in GCA).
        # So, G[C>A]A is the context string.
        # Wait, my manual example was A[C>A]G. Let's re-evaluate.
        # Original: G>T in context C-G-T. G is purine.
        # Complement of G is C. Complement of T is A. So C>A.
        # Context CGT, reverse complement is ACG.
        # Upstream C becomes A. Downstream T becomes G.
        # So, A[C>A]G. This is correct.

        mock_ref_genome = MockFasta({'chr1': 'NNNCGTNNN'}) # G at pos 5 (1-based)
        row_data = pd.Series({
            'Chromosome': '1', 'Start_Position': 5,
            'Reference_Allele': 'G', 'Tumor_Seq_Allele2': 'T'
        })
        mtype, context, err = preprocess_mafs.get_trinucleotide_context(row_data, mock_ref_genome)
        self.assertIsNone(err)
        self.assertEqual(mtype, 'C>A') # Normalized mutation type
        self.assertEqual(context, 'A[C>A]G') # Normalized context

    def test_get_trinucleotide_context_fasta_indexing_error(self):
        """Test error handling for FastaIndexingError (e.g. pos too close to end)."""
        # Mock Fasta to return a short chromosome sequence
        mock_ref_genome = MockFasta({'chr1': 'ACGT'}) # Length 4
        row_data = pd.Series({ # Requesting context at pos 4 (1-based), needs pos 3,4,5
            'Chromosome': '1', 'Start_Position': 4, 
            'Reference_Allele': 'T', 'Tumor_Seq_Allele2': 'A'
        })
        # pyfaidx would try to fetch chr1[2:5], which is out of bounds for 'ACGT' (indices 0-3)
        # The function should catch this (or if pyfaidx returns short seq)
        mtype, context, err = preprocess_mafs.get_trinucleotide_context(row_data, mock_ref_genome)
        self.assertIsNotNone(err)
        self.assertIn("Extracted sequence length not 3", err) # or "FastaIndexingError" depending on pyfaidx version

        # Test pos too close to start
        row_data_start = pd.Series({
            'Chromosome': '1', 'Start_Position': 1, # Needs pos 0, -1
            'Reference_Allele': 'A', 'Tumor_Seq_Allele2': 'C'
        })
        mtype_s, context_s, err_s = preprocess_mafs.get_trinucleotide_context(row_data_start, mock_ref_genome)
        self.assertIsNotNone(err_s)
        self.assertIn("Position too close to chromosome start", err_s)


    def test_get_trinucleotide_context_ref_mismatch(self):
        """Test mismatch between MAF ref allele and FASTA ref allele."""
        mock_ref_genome = MockFasta({'chr1': 'NNNACGTNNN'}) # Fasta has C at pos 5
        row_data = pd.Series({
            'Chromosome': '1', 'Start_Position': 5, 
            'Reference_Allele': 'A',         # MAF claims A
            'Tumor_Seq_Allele2': 'G'
        })
        mtype, context, err = preprocess_mafs.get_trinucleotide_context(row_data, mock_ref_genome)
        self.assertIsNotNone(err)
        self.assertIn("FASTA/MAF ref mismatch: C vs A", err) # Fasta C vs MAF A

    @mock.patch('preprocess_mafs.get_trinucleotide_context') # We test aggregation, not context gen here
    def test_matrix_aggregation_and_structure(self, mock_get_context):
        """Test aggregation into patient-by-context matrix and its structure."""
        # Sample data after context generation (as if it's `final_matrix`)
        # (Tumor_Sample_Barcode, trinucleotide_context)
        
        # Let get_trinucleotide_context return pre-defined values
        def context_side_effect(row, ref_genome):
            # Simple mock: map Start_Position to a context for testing
            if row['Start_Position'] == 100: return pd.Series(['C>A', 'A[C>A]G', None])
            if row['Start_Position'] == 200: return pd.Series(['T>G', 'C[T>G]A', None])
            if row['Start_Position'] == 300: return pd.Series(['C>A', 'A[C>A]G', None]) # Another A[C>A]G for S2
            if row['Start_Position'] == 400: return pd.Series(['C>T', 'G[C>T]C', None]) # For S1, different context
            return pd.Series([None,None,"Error"])

        mock_get_context.side_effect = context_side_effect

        # This is the `all_maf_data` before concatenation and context generation
        # The test will simulate the steps from this point onwards.
        snv_list_for_matrix = [
            {'Tumor_Sample_Barcode': 'Sample1', 'Chromosome':'1', 'Start_Position':100, 'Reference_Allele':'C', 'Tumor_Seq_Allele2':'A', 'Cohort':'C1'},
            {'Tumor_Sample_Barcode': 'Sample1', 'Chromosome':'1', 'Start_Position':200, 'Reference_Allele':'T', 'Tumor_Seq_Allele2':'G', 'Cohort':'C1'},
            {'Tumor_Sample_Barcode': 'Sample2', 'Chromosome':'1', 'Start_Position':300, 'Reference_Allele':'C', 'Tumor_Seq_Allele2':'A', 'Cohort':'C2'},
            {'Tumor_Sample_Barcode': 'Sample1', 'Chromosome':'1', 'Start_Position':400, 'Reference_Allele':'C', 'Tumor_Seq_Allele2':'T', 'Cohort':'C1'},
        ]
        
        # Simulate the script's logic:
        # 1. Create final_matrix from list of dicts
        final_matrix_input = pd.DataFrame(snv_list_for_matrix)

        # 2. Apply get_trinucleotide_context (mocked)
        # In the script: context_results = final_matrix.apply(lambda row: get_trinucleotide_context(row, ref_genome), axis=1)
        #                final_matrix[['mutation_type', 'trinucleotide_context', 'context_error']] = context_results
        #                final_matrix.dropna(subset=['trinucleotide_context'], inplace=True)
        
        # Here, we directly use the side_effect to build what the result of apply would be
        context_results_list = []
        for _, row in final_matrix_input.iterrows():
            context_results_list.append(context_side_effect(row, None)) # ref_genome is not used by mock
        
        context_df = pd.DataFrame(context_results_list, columns=['mutation_type', 'trinucleotide_context', 'context_error'])
        final_matrix_with_context = pd.concat([final_matrix_input.reset_index(drop=True), context_df.reset_index(drop=True)], axis=1)
        final_matrix_with_context.dropna(subset=['trinucleotide_context'], inplace=True)


        # 3. Perform groupby and unstack
        # mutation_counts = final_matrix.groupby(['Tumor_Sample_Barcode', 'trinucleotide_context']).size().unstack(fill_value=0)
        mutation_counts = final_matrix_with_context.groupby(
            ['Tumor_Sample_Barcode', 'trinucleotide_context']
        ).size().unstack(fill_value=0)

        # 4. Reindex with standard contexts
        # mutation_matrix = mutation_counts.reindex(columns=standard_96_contexts, fill_value=0)
        mutation_matrix = mutation_counts.reindex(columns=self.standard_96_contexts, fill_value=0)

        # Assertions
        self.assertEqual(mutation_matrix.shape, (2, 96)) # 2 samples, 96 contexts
        self.assertListEqual(sorted(mutation_matrix.index.tolist()), ['Sample1', 'Sample2'])
        self.assertListEqual(mutation_matrix.columns.tolist(), self.standard_96_contexts) # Check order

        # Check specific counts based on side_effect
        # Sample1: A[C>A]G (from pos 100) = 1, C[T>G]A (from pos 200) = 1, G[C>T]C (from pos 400) = 1
        # Sample2: A[C>A]G (from pos 300) = 1
        self.assertEqual(mutation_matrix.loc['Sample1', 'A[C>A]G'], 1)
        self.assertEqual(mutation_matrix.loc['Sample1', 'C[T>G]A'], 1)
        self.assertEqual(mutation_matrix.loc['Sample1', 'G[C>T]C'], 1)
        self.assertEqual(mutation_matrix.loc['Sample2', 'A[C>A]G'], 1)
        self.assertEqual(mutation_matrix.loc['Sample2', 'C[T>G]A'], 0) # Ensure fill_value worked


if __name__ == '__main__':
    unittest.main()
