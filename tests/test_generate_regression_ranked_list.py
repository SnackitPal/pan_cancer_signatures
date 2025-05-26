import unittest
from unittest.mock import patch, MagicMock, mock_open, call
import pandas as pd
import numpy as np
import argparse
import os
import sys
import io # For capturing stderr
import tempfile # For temporary MAF files

# Adjust sys.path to allow direct import of the script under test
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import functions from the script to be tested
from scripts import generate_regression_ranked_list # Import the module itself
from scripts.generate_regression_ranked_list import (
    load_and_prepare_metadata, build_binary_mutation_matrix, 
    perform_gene_wise_regression, main as regression_main, SAMPLE_ID_COL
)


class TestGenerateRegressionRankedList(unittest.TestCase):
    """Test suite for generate_regression_ranked_list.py script."""

    def setUp(self):
        """Common setup for tests, e.g., creating mock args."""
        self.mock_args = argparse.Namespace(
            exposures_file="dummy_exposures.tsv",
            sample_map_file="dummy_sample_map.tsv",
            tmb_file="dummy_tmb.tsv",
            maf_input_dir="dummy_maf_dir",
            target_cohort="TCGA-TEST",
            target_signature_column="Signature_1",
            tmb_column_name="TMB_per_Mb",
            min_mutations_per_gene=3,
            output_ranked_gene_file="dummy_output_ranked.tsv"
        )

    def _create_mock_df(self, data_dict, index_name=None):
        df = pd.DataFrame(data_dict)
        if index_name:
            df = df.set_index(index_name)
        return df

    def test_argument_parser(self):
        """Tests the argument parsing functionality (condensed)."""
        parser = generate_regression_ranked_list.create_parser()
        args_list_all = ['--exposures_file', 'e.tsv', '--sample_map_file', 'sm.tsv', '--tmb_file', 't.tsv',
                         '--maf_input_dir', 'm/', '--target_cohort', 'C', '--target_signature_column', 'S',
                         '--tmb_column_name', 'T', '--min_mutations_per_gene', '5', '--output_ranked_gene_file', 'o.tsv']
        args = parser.parse_args(args_list_all)
        self.assertEqual(args.min_mutations_per_gene, 5)
        with self.assertRaises(SystemExit): parser.parse_args([])


    # --- Tests for load_and_prepare_metadata ---
    @patch('scripts.generate_regression_ranked_list.pd.read_csv')
    def test_load_prepare_metadata_success(self, mock_read_csv):
        mock_exp_data = {SAMPLE_ID_COL: ['s1', 's2', 's3'], 'Signature_1': [0.1, 0.5, 0.9], 'OtherSig': [0.2,0.3,0.4]}
        mock_map_data = {SAMPLE_ID_COL: ['s1', 's2', 's3', 's4'], 'Cohort': ['TCGA-TEST', 'TCGA-OTHER', 'TCGA-TEST', 'TCGA-TEST']}
        mock_tmb_data = {SAMPLE_ID_COL: ['s1', 's2', 's3'], 'TMB_per_Mb': [5, 10, 15], 'RawTMB': [50,100,150]}
        
        # Configure side_effect for multiple calls to read_csv
        mock_read_csv.side_effect = [
            self._create_mock_df(mock_exp_data, index_name=SAMPLE_ID_COL),
            self._create_mock_df(mock_map_data, index_name=SAMPLE_ID_COL),
            self._create_mock_df(mock_tmb_data, index_name=SAMPLE_ID_COL)
        ]
        
        result_df = load_and_prepare_metadata(self.mock_args)
        
        self.assertEqual(len(result_df), 2) # s1, s3 are in TCGA-TEST and present in all files
        self.assertListEqual(sorted(result_df.index.tolist()), ['s1', 's3'])
        self.assertIn('Signature_1', result_df.columns)
        self.assertIn('TMB_per_Mb', result_df.columns)
        self.assertIn('Cohort', result_df.columns)
        self.assertTrue(all(result_df['Cohort'] == 'TCGA-TEST'))

    @patch('scripts.generate_regression_ranked_list.pd.read_csv', side_effect=FileNotFoundError)
    def test_load_prepare_metadata_file_not_found(self, mock_read_csv):
        with self.assertRaises(SystemExit): # Script exits on FileNotFoundError
            load_and_prepare_metadata(self.mock_args)

    @patch('scripts.generate_regression_ranked_list.pd.read_csv')
    def test_load_prepare_metadata_missing_cols(self, mock_read_csv):
        mock_exp_data = {SAMPLE_ID_COL: ['s1'], self.mock_args.target_signature_column: [0.1]}
        mock_map_data = {SAMPLE_ID_COL: ['s1']} # Missing 'Cohort'
        mock_tmb_data = {SAMPLE_ID_COL: ['s1'], self.mock_args.tmb_column_name: [5]}
        mock_read_csv.side_effect = [
            self._create_mock_df(mock_exp_data, index_name=SAMPLE_ID_COL),
            self._create_mock_df(mock_map_data, index_name=SAMPLE_ID_COL),
            self._create_mock_df(mock_tmb_data, index_name=SAMPLE_ID_COL)
        ]
        with self.assertRaises(SystemExit): # Script exits on missing 'Cohort'
            load_and_prepare_metadata(self.mock_args)
            
    @patch('scripts.generate_regression_ranked_list.pd.read_csv')
    def test_load_prepare_metadata_empty_after_merge_or_filter(self, mock_read_csv):
        # Test empty after cohort filter
        mock_exp_data = {SAMPLE_ID_COL: ['s1'], self.mock_args.target_signature_column: [0.1]}
        mock_map_data = {SAMPLE_ID_COL: ['s1'], 'Cohort': ['TCGA-OTHER']} # No target cohort samples
        mock_tmb_data = {SAMPLE_ID_COL: ['s1'], self.mock_args.tmb_column_name: [5]}
        mock_read_csv.side_effect = [
             self._create_mock_df(mock_exp_data, index_name=SAMPLE_ID_COL),
             self._create_mock_df(mock_map_data, index_name=SAMPLE_ID_COL),
             self._create_mock_df(mock_tmb_data, index_name=SAMPLE_ID_COL)
        ]
        with self.assertRaises(SystemExit):
            load_and_prepare_metadata(self.mock_args)


    # --- Test build_binary_mutation_matrix ---
    @patch('scripts.generate_regression_ranked_list.glob.glob')
    @patch('scripts.generate_regression_ranked_list.pd.read_csv')
    @patch('scripts.generate_regression_ranked_list.os.path.isdir', return_value=True)
    def test_build_binary_mutation_matrix_success(self, mock_isdir, mock_pd_read_csv, mock_glob):
        mock_metadata = self._create_mock_df({SAMPLE_ID_COL: ['S1', 'S2', 'S3']}, index_name=SAMPLE_ID_COL)
        
        maf_data = {
            'Tumor_Sample_Barcode': ['S1', 'S1', 'S2', 'S3', 'S1', 'S2'],
            'Variant_Type': ['SNP', 'SNP', 'SNP', 'SNP', 'INDEL', 'SNP'],
            'Reference_Allele': ['A', 'C', 'G', 'T', 'A', 'C'],
            'Tumor_Seq_Allele2': ['T', 'G', 'A', 'C', 'AT', 'A'],
            'Hugo_Symbol': ['GENE1', 'GENE2', 'GENE1', 'GENE3', 'GENE1', 'GENE4']
        } # S1: GENE1, GENE2; S2: GENE1, GENE4; S3: GENE3
        mock_maf_df = pd.DataFrame(maf_data)
        
        mock_glob.return_value = ['dummy_maf_dir/TCGA-TEST/file1.maf.gz']
        mock_pd_read_csv.return_value = mock_maf_df
        
        result_matrix = build_binary_mutation_matrix(self.mock_args, mock_metadata)
        
        self.assertEqual(result_matrix.shape, (3, 4)) # 3 samples, 4 unique genes
        self.assertListEqual(sorted(result_matrix.columns.tolist()), sorted(['GENE1', 'GENE2', 'GENE3', 'GENE4']))
        self.assertListEqual(sorted(result_matrix.index.tolist()), sorted(['S1', 'S2', 'S3']))
        self.assertEqual(result_matrix.loc['S1', 'GENE1'], 1)
        self.assertEqual(result_matrix.loc['S1', 'GENE2'], 1)
        self.assertEqual(result_matrix.loc['S1', 'GENE4'], 0)
        self.assertEqual(result_matrix.loc['S2', 'GENE1'], 1)
        self.assertEqual(result_matrix.loc['S2', 'GENE4'], 1)
        self.assertEqual(result_matrix.loc['S3', 'GENE3'], 1)

    @patch('scripts.generate_regression_ranked_list.glob.glob', return_value=[]) # No MAF files
    @patch('scripts.generate_regression_ranked_list.os.path.isdir', return_value=True)
    def test_build_binary_mutation_matrix_no_mafs(self, mock_isdir, mock_glob):
        mock_metadata = self._create_mock_df({SAMPLE_ID_COL: ['S1']}, index_name=SAMPLE_ID_COL)
        result_matrix = build_binary_mutation_matrix(self.mock_args, mock_metadata)
        self.assertTrue(result_matrix.empty or result_matrix.shape[1] == 0) # Samples as index, no gene columns
        self.assertListEqual(result_matrix.index.tolist(), ['S1'])


    # --- Test perform_gene_wise_regression (Mocking Statsmodels) ---
    @patch('scripts.generate_regression_ranked_list.sm.Logit')
    @patch('scripts.generate_regression_ranked_list.sm.add_constant')
    def test_perform_regression_success_and_rank_metric(self, mock_add_constant, mock_Logit_class):
        mock_metadata = self._create_mock_df({
            SAMPLE_ID_COL: ['S1', 'S2', 'S3', 'S4', 'S5'],
            self.mock_args.target_signature_column: [0.1, 0.2, 0.05, 0.5, 0.3],
            self.mock_args.tmb_column_name: [5, 10, 8, 20, 12]
        }, index_name=SAMPLE_ID_COL)
        
        mock_mutation_matrix = self._create_mock_df({
            'GENE_A': [1,0,1,0,1], # 3 mutations
            'GENE_B': [0,1,0,1,1], # 3 mutations
            'GENE_C': [1,0,0,0,0]  # 1 mutation (skipped)
        }, index_name=SAMPLE_ID_COL)
        mock_mutation_matrix.index = pd.Index(['S1', 'S2', 'S3', 'S4', 'S5'], name=SAMPLE_ID_COL)

        # Configure mock Logit and fit results
        mock_logit_instance = MagicMock()
        mock_Logit_class.return_value = mock_logit_instance
        mock_fit_results_geneA = MagicMock()
        mock_fit_results_geneA.params = pd.Series({'const': -0.5, self.mock_args.target_signature_column: 1.5, self.mock_args.tmb_column_name: 0.05})
        mock_fit_results_geneA.pvalues = pd.Series({'const': 0.1, self.mock_args.target_signature_column: 0.04, self.mock_args.tmb_column_name: 0.5})
        mock_fit_results_geneA.tvalues = pd.Series({'const': -1.0, self.mock_args.target_signature_column: 2.5, self.mock_args.tmb_column_name: 0.8}) # z-scores
        
        mock_fit_results_geneB = MagicMock() # For GENE_B, simulate negative coefficient and p=0
        mock_fit_results_geneB.params = pd.Series({'const': 0.2, self.mock_args.target_signature_column: -2.0, self.mock_args.tmb_column_name: -0.1})
        mock_fit_results_geneB.pvalues = pd.Series({'const': 0.3, self.mock_args.target_signature_column: 0.0, self.mock_args.tmb_column_name: 0.2})
        mock_fit_results_geneB.tvalues = pd.Series({'const': 0.5, self.mock_args.target_signature_column: -3.0, self.mock_args.tmb_column_name: -1.5})
        
        mock_logit_instance.fit.side_effect = [mock_fit_results_geneA, mock_fit_results_geneB]
        # Ensure add_constant returns a DataFrame that can be used by Logit
        mock_add_constant.side_effect = lambda x, prepend: pd.concat([pd.Series(np.ones(len(x)), index=x.index, name='const'), x], axis=1)


        results = perform_gene_wise_regression(mock_metadata, mock_mutation_matrix, self.mock_args)
        
        self.assertEqual(len(results), 2) # GENE_C skipped
        self.assertEqual(mock_add_constant.call_count, 2)
        self.assertEqual(mock_Logit_class.call_count, 2)
        self.assertEqual(mock_logit_instance.fit.call_count, 2)
        
        # GENE_A results
        res_A = next(r for r in results if r['Gene'] == 'GENE_A')
        self.assertEqual(res_A['Status'], 'Success')
        self.assertAlmostEqual(res_A['Coefficient_Signature'], 1.5)
        self.assertAlmostEqual(res_A['PValue_Signature'], 0.04)
        self.assertAlmostEqual(res_A['Z_Score_Signature'], 2.5)
        expected_rank_A = np.sign(1.5) * -np.log10(0.04)
        self.assertAlmostEqual(res_A['RankMetric'], expected_rank_A)

        # GENE_B results (p-value of 0)
        res_B = next(r for r in results if r['Gene'] == 'GENE_B')
        self.assertEqual(res_B['Status'], 'Success')
        self.assertAlmostEqual(res_B['Coefficient_Signature'], -2.0)
        self.assertAlmostEqual(res_B['PValue_Signature'], 0.0)
        expected_rank_B = np.sign(-2.0) * -np.log10(np.nextafter(0,1)) # Check against capped p-value
        self.assertAlmostEqual(res_B['RankMetric'], expected_rank_B)

    @patch('scripts.generate_regression_ranked_list.sm.Logit')
    @patch('scripts.generate_regression_ranked_list.sm.add_constant')
    @patch('sys.stderr', new_callable=io.StringIO)
    def test_perform_regression_fit_error(self, mock_stderr, mock_add_constant, mock_Logit_class):
        mock_metadata = self._create_mock_df({SAMPLE_ID_COL: ['S1','S2','S3'], self.mock_args.target_signature_column: [0.1,0.2,0.3], self.mock_args.tmb_column_name: [1,2,3]}, index_name=SAMPLE_ID_COL)
        mock_mutation_matrix = self._create_mock_df({'GENE_FAIL': [1,0,1]}, index_name=SAMPLE_ID_COL)
        mock_mutation_matrix.index = pd.Index(['S1','S2','S3'], name=SAMPLE_ID_COL)


        mock_logit_instance = MagicMock()
        mock_Logit_class.return_value = mock_logit_instance
        mock_logit_instance.fit.side_effect = generate_regression_ranked_list.PerfectSeparationError("Test Error")
        mock_add_constant.side_effect = lambda x, prepend: pd.concat([pd.Series(np.ones(len(x)), index=x.index, name='const'), x], axis=1)


        results = perform_gene_wise_regression(mock_metadata, mock_mutation_matrix, self.mock_args)
        self.assertEqual(len(results), 1)
        res_fail = results[0]
        self.assertEqual(res_fail['Gene'], 'GENE_FAIL')
        self.assertEqual(res_fail['Status'], 'Failed_Fit_PerfectSeparationError')
        self.assertEqual(res_fail['RankMetric'], 0.0)
        self.assertTrue(np.isnan(res_fail['PValue_Signature']))
        # The warning print is commented out in the script, so stderr won't have it.


    # --- Test Output Generation (via main) ---
    @patch('scripts.generate_regression_ranked_list.load_and_prepare_metadata')
    @patch('scripts.generate_regression_ranked_list.build_binary_mutation_matrix')
    @patch('scripts.generate_regression_ranked_list.perform_gene_wise_regression')
    @patch('pandas.DataFrame.to_csv') # Mock the final output step
    @patch('os.makedirs') # Mock directory creation
    def test_main_output_generation(self, mock_makedirs, mock_to_csv, mock_perform_regression, 
                                    mock_build_matrix, mock_load_meta):
        # Setup mock return values for each major step
        mock_load_meta.return_value = pd.DataFrame({SAMPLE_ID_COL: ['s1']}, index=pd.Index(['s1'], name=SAMPLE_ID_COL)) # Minimal valid metadata
        mock_build_matrix.return_value = pd.DataFrame({'GENE1': [1]}, index=pd.Index(['s1'], name=SAMPLE_ID_COL)) # Minimal valid mut matrix
        
        mock_regression_results = [
            {'Gene': 'GENE1', 'RankMetric': 2.5, 'Status': 'Success', 'PValue_Signature': 0.003, 'Coefficient_Signature': 1, 'Z_Score_Signature': 2, 'Num_Mutations': 5},
            {'Gene': 'GENE2', 'RankMetric': -1.5, 'Status': 'Success', 'PValue_Signature': 0.03, 'Coefficient_Signature': -1, 'Z_Score_Signature': -2, 'Num_Mutations': 6}
        ]
        mock_perform_regression.return_value = mock_regression_results

        test_args_list = [
            '--exposures_file', 'e.tsv', '--sample_map_file', 'sm.tsv', '--tmb_file', 't.tsv',
            '--maf_input_dir', 'm/', '--target_cohort', 'C', '--target_signature_column', 'S',
            '--tmb_column_name', 'T', '--output_ranked_gene_file', self.mock_args.output_ranked_gene_file
        ]
        
        regression_main(test_args_list)

        self.assertEqual(mock_to_csv.call_count, 2) # Called for full stats and for ranked list
        
        # Verify call for the ranked list file
        ranked_list_call = None
        full_stats_call = None
        for c in mock_to_csv.call_args_list:
            if c[0][0] == self.mock_args.output_ranked_gene_file:
                ranked_list_call = c
            elif "_full_stats.tsv" in c[0][0]:
                full_stats_call = c
        
        self.assertIsNotNone(ranked_list_call, "to_csv not called for ranked list file")
        self.assertEqual(ranked_list_call[1]['sep'], '\t')
        self.assertEqual(ranked_list_call[1]['index'], False)
        self.assertEqual(ranked_list_call[1]['header'], False)
        
        df_ranked = ranked_list_call[0][0] # The DataFrame passed to to_csv
        self.assertListEqual(list(df_ranked.columns), ['Gene', 'RankMetric'])
        self.assertEqual(df_ranked.iloc[0]['Gene'], 'GENE1') # GENE1 has higher RankMetric
        self.assertEqual(df_ranked.iloc[1]['Gene'], 'GENE2')

        # Verify call for the full stats file
        self.assertIsNotNone(full_stats_call, "to_csv not called for full stats file")
        df_full = full_stats_call[0][0]
        self.assertIn('Status', df_full.columns)


    @patch('scripts.generate_regression_ranked_list.load_and_prepare_metadata')
    @patch('scripts.generate_regression_ranked_list.build_binary_mutation_matrix')
    @patch('scripts.generate_regression_ranked_list.perform_gene_wise_regression', return_value=[]) # Empty results
    @patch('pandas.DataFrame.to_csv')
    def test_main_output_empty_regression_results(self, mock_to_csv, mock_perform_regression, 
                                                 mock_build_matrix, mock_load_meta):
        mock_load_meta.return_value = pd.DataFrame({SAMPLE_ID_COL: ['s1']}, index=pd.Index(['s1'], name=SAMPLE_ID_COL))
        mock_build_matrix.return_value = pd.DataFrame({'GENE1': [1]}, index=pd.Index(['s1'], name=SAMPLE_ID_COL))
        
        test_args_list = [
            '--exposures_file', 'e.tsv', '--sample_map_file', 'sm.tsv', '--tmb_file', 't.tsv',
            '--maf_input_dir', 'm/', '--target_cohort', 'C', '--target_signature_column', 'S',
            '--tmb_column_name', 'T', '--output_ranked_gene_file', 'out.tsv'
        ]
        regression_main(test_args_list)
        # to_csv for full stats might still be called with an empty DataFrame (0 rows),
        # but the ranked list specific call should not happen if results are empty
        # The script logic is: if not regression_results_list: print message; return
        # So, if that return happens, to_csv for ranked list won't be called.
        # The full_stats.tsv IS written even if empty (as per current script)
        # If that's the case, one call to to_csv for the empty full_stats.
        self.assertTrue(mock_to_csv.call_count <= 1) # Can be 0 or 1 depending on exact empty df handling for full_stats.
                                                     # Given current script, it's 1 for full_stats.


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
