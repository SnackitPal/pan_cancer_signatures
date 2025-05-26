import unittest
from unittest.mock import patch, MagicMock, mock_open, call
import pandas as pd
import numpy as np
import argparse
import os
import sys
import io # For capturing stderr
import tempfile # For temporary output file

# Adjust sys.path to allow direct import of the script under test
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import functions from the script to be tested
from scripts import calculate_tmb # Import the module itself

class TestCalculateTmb(unittest.TestCase):
    """Test suite for calculate_tmb.py script."""

    def _create_mock_args(self, maf_dir="dummy_maf_dir", cohorts="C1,C2", exome_size=30.0, output_file="dummy_tmb.tsv"):
        return argparse.Namespace(
            maf_input_dir=maf_dir,
            cohort_list=cohorts,
            exome_size_mb=exome_size,
            output_tmb_file=output_file
        )

    def test_argument_parser(self):
        """Tests the argument parsing functionality."""
        parser = calculate_tmb.create_parser()
        args_list_all = [
            '--maf_input_dir', 'data/mafs', '--cohort_list', 'TCGA-LUAD,TCGA-BRCA',
            '--exome_size_mb', '35.5', '--output_tmb_file', 'results/tmb.tsv'
        ]
        args = parser.parse_args(args_list_all)
        self.assertEqual(args.maf_input_dir, 'data/mafs')
        self.assertEqual(args.exome_size_mb, 35.5)

        args_list_default_exome = [
            '--maf_input_dir', 'data/mafs', '--cohort_list', 'TCGA-LUAD',
            '--output_tmb_file', 'results/tmb.tsv'
        ]
        args_defaults = parser.parse_args(args_list_default_exome)
        self.assertEqual(args_defaults.exome_size_mb, 30.0)

        with self.assertRaises(SystemExit): parser.parse_args([])


    @patch('scripts.calculate_tmb._run_tmb_pipeline') 
    def test_main_cohort_list_processing_valid(self, mock_run_pipeline):
        test_args = ['--maf_input_dir', 'd', '--cohort_list', 'C1, C2', '--output_tmb_file', 'f.tsv']
        calculate_tmb.main(test_args)
        mock_run_pipeline.assert_called_once()
        processed_cohorts_passed = mock_run_pipeline.call_args[0][1]
        self.assertEqual(processed_cohorts_passed, ['C1', 'C2'])

    @patch('scripts.calculate_tmb._run_tmb_pipeline')
    @patch('sys.exit') 
    @patch('sys.stderr', new_callable=io.StringIO) 
    def test_main_cohort_list_processing_empty_string(self, mock_stderr, mock_sys_exit, mock_run_pipeline):
        test_args_empty_cohort = ['--maf_input_dir', 'd', '--cohort_list', '', '--output_tmb_file', 'f.tsv']
        calculate_tmb.main(test_args_empty_cohort)
        mock_sys_exit.assert_called_once_with(1)
        self.assertIn("Error: Cohort list is empty or invalid.", mock_stderr.getvalue())
        mock_run_pipeline.assert_not_called()

    # --- Helper Function ---
    def _create_mock_maf_df(self, data_dict):
        """Creates a mock MAF DataFrame from a dictionary."""
        return pd.DataFrame(data_dict)

    # --- Test SNV Counting Logic ---
    def test_snv_counting_logic(self):
        """Tests SNV identification and counting for various scenarios in a single MAF."""
        sample_snv_counts_dict = {}
        mock_maf_data = {
            'Tumor_Sample_Barcode': ['S1', 'S1', 'S1', 'S1', 'S1', 'S1', 'S1', 'S2', 'S2', 'S3', 'S3'],
            'Variant_Type':       ['SNP', 'SNP', 'INDEL', 'SNP', 'SNP', 'SNP', 'SNP', 'SNP', 'SNP', 'SNP', 'SNP'],
            'Reference_Allele':   ['A',   'C',   'G',     'T',   'A',   'N',   'AC',  'G',   'T',   'C',  'A'],
            'Tumor_Seq_Allele2':  ['T',   'G',   'A',     'A',   'A',   'C',   'T',   'C',   'T',   'G',  'G']
        }
        # Expected for S1: A->T, C->G, T->A. ('A'=='A' is not SNV, 'N' is invalid, 'AC' is not SNV) -> 3
        # Expected for S2: G->C, T->T (is not SNV) -> 1
        # Expected for S3: C->G, A->G -> 2
        mock_maf_df = self._create_mock_maf_df(mock_maf_data)

        # To test process_maf_file_for_snvs, we need to mock its reading of the file
        # by making it operate directly on the DataFrame we created.
        # We can achieve this by patching pd.read_csv if process_maf_file_for_snvs uses it,
        # or by refactoring process_maf_file_for_snvs to accept a DataFrame (preferred but not current structure).
        # For now, let's assume we test the logic more directly or adapt the file-based function.
        # The current `process_maf_file_for_snvs` takes a path.
        # We will write this mock_maf_df to a temporary file.
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".maf") as tmp_maf:
            mock_maf_df.to_csv(tmp_maf.name, sep='\t', index=False)
            tmp_maf_path = tmp_maf.name
        
        calculate_tmb.process_maf_file_for_snvs(tmp_maf_path, sample_snv_counts_dict)
        os.unlink(tmp_maf_path) # Clean up

        self.assertEqual(sample_snv_counts_dict.get('S1', 0), 3)
        self.assertEqual(sample_snv_counts_dict.get('S2', 0), 1)
        self.assertEqual(sample_snv_counts_dict.get('S3', 0), 2)
        self.assertNotIn('S_NonExistent', sample_snv_counts_dict)


    # --- Test Sample Aggregation ---
    @patch('scripts.calculate_tmb.glob.glob')
    @patch('scripts.calculate_tmb.pd.read_csv')
    @patch('scripts.calculate_tmb.os.path.isdir', return_value=True) # Assume cohort dirs exist
    def test_sample_aggregation_across_mafs(self, mock_isdir, mock_pd_read_csv, mock_glob_glob):
        """Tests aggregation of SNV counts for the same sample across multiple MAF files."""
        args = self._create_mock_args(cohorts="COH1") # Test with one cohort for simplicity
        
        maf_data1 = {
            'Tumor_Sample_Barcode': ['SampleA', 'SampleB'], 'Variant_Type': ['SNP', 'SNP'],
            'Reference_Allele':   ['A', 'C'], 'Tumor_Seq_Allele2': ['T', 'G']
        } # SampleA: 1, SampleB: 1
        maf_data2 = {
            'Tumor_Sample_Barcode': ['SampleA', 'SampleC'], 'Variant_Type': ['SNP', 'SNP'],
            'Reference_Allele':   ['G', 'T'], 'Tumor_Seq_Allele2': ['A', 'C']
        } # SampleA: 1, SampleC: 1
        
        df_maf1 = self._create_mock_maf_df(maf_data1)
        df_maf2 = self._create_mock_maf_df(maf_data2)

        mock_glob_glob.return_value = ['dummy_maf_dir/COH1/maf1.maf.gz', 'dummy_maf_dir/COH1/maf2.maf.gz']
        mock_pd_read_csv.side_effect = [df_maf1, df_maf2]
        
        # We test the _run_tmb_pipeline function as it contains the aggregation logic
        # We need to capture sample_snv_counts, which is internal to it.
        # So, we'll patch the parts after SNV counting (DataFrame creation and saving)
        with patch('pandas.DataFrame.to_csv') as mock_to_csv, \
             patch('os.makedirs'): # Mock makedirs as well for saving part
            
            # The structure of _run_tmb_pipeline means we can't easily get sample_snv_counts
            # without it trying to save. The alternative is to test main and mock _run_tmb_pipeline
            # but for this specific test, we want to test *inside* _run_tmb_pipeline's loop.
            # A slight refactor of _run_tmb_pipeline to return sample_snv_counts would be ideal.
            # For now, we'll let it run and inspect the DataFrame passed to to_csv.

            calculate_tmb._run_tmb_pipeline(args, ["COH1"])

            mock_to_csv.assert_called_once()
            df_passed_to_csv = mock_to_csv.call_args[0][0] # First arg of first call
            
            # Expected counts: SampleA: 2, SampleB: 1, SampleC: 1
            # The df will have 'Tumor_Sample_Barcode' and 'Total_SNVs' (and maybe TMB)
            self.assertEqual(len(df_passed_to_csv), 3)
            self.assertTrue('SampleA' in df_passed_to_csv['Tumor_Sample_Barcode'].values)
            self.assertEqual(df_passed_to_csv[df_passed_to_csv['Tumor_Sample_Barcode'] == 'SampleA']['Total_SNVs'].iloc[0], 2)
            self.assertEqual(df_passed_to_csv[df_passed_to_csv['Tumor_Sample_Barcode'] == 'SampleB']['Total_SNVs'].iloc[0], 1)
            self.assertEqual(df_passed_to_csv[df_passed_to_csv['Tumor_Sample_Barcode'] == 'SampleC']['Total_SNVs'].iloc[0], 1)


    # --- Test TMB Normalization & DataFrame Output ---
    @patch('scripts.calculate_tmb.glob.glob', return_value=[]) # No MAFs to process, focus on output stage
    @patch('scripts.calculate_tmb.os.path.isdir', return_value=True)
    @patch('pandas.DataFrame.to_csv')
    @patch('os.makedirs') # Mock makedirs for saving
    def test_tmb_normalization_and_output(self, mock_os_makedirs, mock_to_csv, mock_isdir, mock_glob):
        """Tests TMB normalization and DataFrame output logic."""
        
        # --- Case 1: Normalization enabled ---
        args_norm = self._create_mock_args(exome_size=30.0, cohorts="C1") # Use just one cohort for simplicity
        # Manually set sample_snv_counts as if MAF processing happened
        # To do this cleanly, we need to inject it into _run_tmb_pipeline or mock its internal MAF loop
        # For this test, we'll mock the MAF processing part of _run_tmb_pipeline
        with patch('scripts.calculate_tmb.process_maf_file_for_snvs') as mock_process_mafs:
            # This lambda will be called by _run_tmb_pipeline.
            # It needs to populate the sample_snv_counts dict that _run_tmb_pipeline creates.
            def side_effect_populate_counts(maf_path, counts_dict):
                counts_dict['SampleA'] = 60
                counts_dict['SampleB'] = 30
            mock_process_mafs.side_effect = side_effect_populate_counts
            # Need to ensure glob.glob returns something for the loop to run once to call process_maf_file_for_snvs
            mock_glob.return_value = ['dummy_maf_dir/C1/one.maf']


            calculate_tmb._run_tmb_pipeline(args_norm, ["C1"])

        mock_to_csv.assert_called_once()
        df_norm = mock_to_csv.call_args[0][0]
        self.assertIn('Tumor_Sample_Barcode', df_norm.columns)
        self.assertIn('Total_SNVs', df_norm.columns)
        self.assertIn('TMB_mut_per_Mb', df_norm.columns)
        self.assertAlmostEqual(df_norm[df_norm['Tumor_Sample_Barcode'] == 'SampleA']['TMB_mut_per_Mb'].iloc[0], 2.0)
        self.assertAlmostEqual(df_norm[df_norm['Tumor_Sample_Barcode'] == 'SampleB']['TMB_mut_per_Mb'].iloc[0], 1.0)
        mock_to_csv.reset_mock() # Reset for next case

        # --- Case 2: Normalization disabled ---
        args_no_norm = self._create_mock_args(exome_size=0.0, cohorts="C1")
        with patch('scripts.calculate_tmb.process_maf_file_for_snvs') as mock_process_mafs:
            def side_effect_populate_counts_no_norm(maf_path, counts_dict):
                counts_dict['SampleA'] = 60
                counts_dict['SampleB'] = 30
            mock_process_mafs.side_effect = side_effect_populate_counts_no_norm
            mock_glob.return_value = ['dummy_maf_dir/C1/one.maf'] # Ensure loop runs

            calculate_tmb._run_tmb_pipeline(args_no_norm, ["C1"])
        
        mock_to_csv.assert_called_once()
        df_no_norm = mock_to_csv.call_args[0][0]
        self.assertIn('Tumor_Sample_Barcode', df_no_norm.columns)
        self.assertIn('Total_SNVs', df_no_norm.columns)
        self.assertNotIn('TMB_mut_per_Mb', df_no_norm.columns)
        self.assertEqual(df_no_norm[df_no_norm['Tumor_Sample_Barcode'] == 'SampleA']['Total_SNVs'].iloc[0], 60)
        mock_to_csv.reset_mock()

        # --- Case 3: Empty sample_snv_counts ---
        args_empty = self._create_mock_args(cohorts="C1")
        # This time, process_maf_file_for_snvs will not populate sample_snv_counts
        with patch('scripts.calculate_tmb.process_maf_file_for_snvs') as mock_process_mafs_empty:
             mock_glob.return_value = ['dummy_maf_dir/C1/one.maf'] # Ensure loop runs
             # but side_effect of mock_process_mafs_empty is default (None), so dict remains empty
             calculate_tmb._run_tmb_pipeline(args_empty, ["C1"])
        
        mock_to_csv.assert_not_called() # Should not be called if no data


    # --- Test MAF Column Validation ---
    @patch('sys.stderr', new_callable=io.StringIO) # Capture stderr
    def test_maf_column_validation_in_process_maf(self, mock_stderr):
        """Tests that process_maf_file_for_snvs handles missing required columns."""
        sample_snv_counts_dict = {}
        # Missing 'Variant_Type'
        mock_maf_data_missing_col = {
            'Tumor_Sample_Barcode': ['S1'], 'Reference_Allele': ['A'], 'Tumor_Seq_Allele2': ['T']
        }
        mock_maf_df_missing_col = self._create_mock_maf_df(mock_maf_data_missing_col)

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".maf") as tmp_maf:
            mock_maf_df_missing_col.to_csv(tmp_maf.name, sep='\t', index=False)
            tmp_maf_path = tmp_maf.name

        calculate_tmb.process_maf_file_for_snvs(tmp_maf_path, sample_snv_counts_dict)
        os.unlink(tmp_maf_path)

        self.assertEqual(len(sample_snv_counts_dict), 0) # No counts should be added
        self.assertIn("missing required columns: Variant_Type", mock_stderr.getvalue())


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
