import unittest
from unittest.mock import patch, MagicMock, mock_open
import pandas as pd
import numpy as np
import argparse
import sys
import os
from scipy.stats import fisher_exact # For verifying results

# Add script directory to path to allow direct import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts')))

from generate_gsea_ranked_list import (
    create_parser, load_and_filter_data, define_exposure_groups, 
    process_maf_files, create_binary_mutation_matrix, perform_differential_analysis,
    save_ranked_list, ScriptLogicError
)

class TestGenerateGseaRankedList(unittest.TestCase):

    def _create_mock_args(self):
        return argparse.Namespace(
            exposures_file="dummy_exposures.tsv",
            sample_map_file="dummy_sample_map.tsv",
            maf_input_dir="dummy_maf_dir/",
            target_cohort="COHORT_A",
            target_signature_column="Signature_1",
            high_exposure_quantile=0.75,
            low_exposure_quantile=0.25,
            min_group_size=2, 
            output_ranked_gene_file="dummy_output.rnk"
        )

    # --- Helper methods for creating mock DataFrames ---
    def _create_mock_exposures_df(self, sample_ids, signature_name="Signature_1", signature_values=None):
        if signature_values is None:
            signature_values = np.random.rand(len(sample_ids))
        return pd.DataFrame({
            'sample_id': sample_ids,
            signature_name: signature_values
        })

    def _create_mock_sample_map_df(self, sample_ids, cohorts):
        return pd.DataFrame({
            'SampleID': sample_ids,
            'Cohort': cohorts
        })

    def _create_mock_maf_df(self, sample_ids, hugo_symbols, variant_types):
        """
        Creates a mock MAF DataFrame.
        Assumes sample_ids, hugo_symbols, and variant_types are lists of the same length.
        """
        return pd.DataFrame({
            'Tumor_Sample_Barcode': sample_ids,
            'Hugo_Symbol': hugo_symbols,
            'Variant_Type': variant_types
        })

    # --- Tests for argument parsing (already implemented and verified) ---
    def test_argument_parser(self):
        # This test was previously implemented and verified. Keeping it short for brevity.
        parser = create_parser()
        args_list = [
            "--exposures_file", "path/to/exposures.tsv", "--sample_map_file", "path/to/sample_map.tsv",
            "--maf_input_dir", "path/to/maf_dir/", "--target_cohort", "TCGA-LUAD",
            "--target_signature_column", "SBS1", "--high_exposure_quantile", "0.75",
            "--low_exposure_quantile", "0.25", "--min_group_size", "10",
            "--output_ranked_gene_file", "path/to/output.rnk"
        ]
        args = parser.parse_args(args_list)
        self.assertEqual(args.exposures_file, "path/to/exposures.tsv")
        with self.assertRaises(SystemExit):
            parser.parse_args(args_list[:-2]) # Missing last arg

    # --- Tests for load_and_filter_data (previously implemented) ---
    @patch('pandas.read_csv')
    def test_successful_load_merge_filter(self, mock_read_csv):
        args = self._create_mock_args()
        mock_exp_df = self._create_mock_exposures_df(['s1', 's2', 's3', 's4'], args.target_signature_column)
        mock_map_df = self._create_mock_sample_map_df(['s1', 's2', 's3', 's4'], ['COHORT_A', 'COHORT_B', 'COHORT_A', 'COHORT_B'])
        mock_read_csv.side_effect = [mock_exp_df, mock_map_df]
        cohort_df = load_and_filter_data(args)
        self.assertEqual(len(cohort_df), 2)

    # --- Tests for define_exposure_groups (previously implemented) ---
    def test_successful_group_definition(self):
        args = self._create_mock_args()
        cohort_data = [('s1', 0.1, 's1', 'COHORT_A'), ('s2', 0.2, 's2', 'COHORT_A'), ('s3', 0.8, 's3', 'COHORT_A'), ('s4', 0.9, 's4', 'COHORT_A')]
        mock_cohort_df = pd.DataFrame(cohort_data, columns=['sample_id', args.target_signature_column, 'SampleID', 'Cohort'])
        args.low_exposure_quantile = 0.50; args.high_exposure_quantile = 0.50 # ensure 2 per group
        high_group, low_group = define_exposure_groups(mock_cohort_df, args)
        self.assertEqual(len(low_group), 2)
        self.assertEqual(len(high_group), 2)

    # --- Tests for process_maf_files ---
    @patch('generate_gsea_ranked_list.pd.read_csv') # Patching in the context of the imported module
    @patch('generate_gsea_ranked_list.glob.glob')
    def test_successful_maf_processing(self, mock_glob, mock_pd_read_csv):
        args = self._create_mock_args()
        target_samples = ['s1', 's2', 's3_other_cohort'] # s3 is not in target_cohort_samples for this test
        
        mock_glob.return_value = ["dummy_maf_dir/COHORT_A/file1.maf.gz"]
        
        maf_data_s1_g1 = self._create_mock_maf_df(sample_ids=['s1'], hugo_symbols=['GENE1'], variant_types=['SNP'])
        maf_data_s2_g2 = self._create_mock_maf_df(sample_ids=['s2'], hugo_symbols=['GENE2'], variant_types=['SNP'])
        maf_data_s1_g2_indel = self._create_mock_maf_df(sample_ids=['s1'], hugo_symbols=['GENE2'], variant_types=['INDEL']) # Should be ignored
        maf_data_s3_g3 = self._create_mock_maf_df(sample_ids=['s3_other_cohort'], hugo_symbols=['GENE3'], variant_types=['SNP']) # Ignored sample

        # Concatenate them to simulate a single MAF file's content
        simulated_maf_content = pd.concat([maf_data_s1_g1, maf_data_s2_g2, maf_data_s1_g2_indel, maf_data_s3_g3], ignore_index=True)
        mock_pd_read_csv.return_value = simulated_maf_content

        sample_mutations, all_genes = process_maf_files(args, target_samples)

        self.assertEqual(len(sample_mutations), 2) # s1 and s2
        self.assertIn('s1', sample_mutations)
        self.assertEqual(sample_mutations['s1'], {'GENE1'}) # GENE2 was INDEL
        self.assertIn('s2', sample_mutations)
        self.assertEqual(sample_mutations['s2'], {'GENE2'})
        self.assertNotIn('s3_other_cohort', sample_mutations)
        
        self.assertEqual(all_genes, {'GENE1', 'GENE2'}) # GENE3 was from non-target sample, GENE2 (INDEL) ignored

    @patch('generate_gsea_ranked_list.glob.glob')
    def test_no_maf_files_found(self, mock_glob):
        args = self._create_mock_args()
        target_samples = ['s1', 's2']
        mock_glob.return_value = [] # No files found

        sample_mutations, all_genes = process_maf_files(args, target_samples)
        self.assertEqual(len(sample_mutations), 0)
        self.assertEqual(len(all_genes), 0)

    # --- Tests for create_binary_mutation_matrix ---
    def test_standard_matrix_creation(self):
        target_samples = ['s1', 's2', 's3']
        all_genes_set = {'GENE1', 'GENE2', 'GENE3'}
        sample_mutations_dict = {
            's1': {'GENE1', 'GENE2'},
            's2': {'GENE2'},
            # s3 has no mutations in this dict, will be all zeros
        }
        
        matrix = create_binary_mutation_matrix(target_samples, all_genes_set, sample_mutations_dict)
        
        self.assertEqual(matrix.shape, (3, 3)) # 3 samples, 3 genes
        self.assertEqual(matrix.loc['s1', 'GENE1'], 1)
        self.assertEqual(matrix.loc['s1', 'GENE2'], 1)
        self.assertEqual(matrix.loc['s1', 'GENE3'], 0)
        self.assertEqual(matrix.loc['s2', 'GENE1'], 0)
        self.assertEqual(matrix.loc['s2', 'GENE2'], 1)
        self.assertEqual(matrix.loc['s3', 'GENE1'], 0)
        self.assertEqual(matrix.loc['s3', 'GENE2'], 0)
        self.assertEqual(matrix.loc['s3', 'GENE3'], 0)

    def test_empty_sample_mutations_matrix(self):
        target_samples = ['s1', 's2']
        all_genes_set = {'GENE1', 'GENE2'} # Genes exist, but no mutations in them
        sample_mutations_dict = {}
        
        matrix = create_binary_mutation_matrix(target_samples, all_genes_set, sample_mutations_dict)
        self.assertEqual(matrix.shape, (2, 2))
        self.assertTrue((matrix == 0).all().all())

        # Test with no genes at all
        matrix_no_genes = create_binary_mutation_matrix(target_samples, set(), {})
        self.assertEqual(matrix_no_genes.shape, (2,0))


    # --- Tests for perform_differential_analysis ---
    def test_basic_differential_analysis(self):
        # Gene1: mutated in high, not in low
        # Gene2: mutated in low, not in high
        # Gene3: mutated in both
        # Gene4: mutated in neither
        mutation_data = {
            'GENE1': [1, 1, 0, 0, 0, 0], # Mutated in s1, s2 (high); Not in s3, s4 (low)
            'GENE2': [0, 0, 1, 1, 0, 0], # Mutated in s3, s4 (low); Not in s1, s2 (high)
            'GENE3': [1, 0, 1, 0, 0, 0], # Mutated in s1 (high), s3 (low)
            'GENE4': [0, 0, 0, 0, 0, 0], # No mutations
        }
        samples = ['s1_high', 's2_high', 's3_low', 's4_low', 's5_high_nomut', 's6_low_nomut']
        mutation_matrix = pd.DataFrame(mutation_data, index=samples)
        
        high_exposure_ids = {'s1_high', 's2_high', 's5_high_nomut'} # 3 samples
        low_exposure_ids = {'s3_low', 's4_low', 's6_low_nomut'}   # 3 samples
        all_genes_set = {'GENE1', 'GENE2', 'GENE3', 'GENE4'}

        results_df = perform_differential_analysis(mutation_matrix, high_exposure_ids, low_exposure_ids, all_genes_set)
        
        self.assertEqual(len(results_df), 4)
        
        # GENE1: High (2/3 mut), Low (0/3 mut)
        gene1_res = results_df[results_df['Gene'] == 'GENE1'].iloc[0]
        # table = [[2, 1], [0, 3]] -> OR = inf, p-val (calculated by scipy)
        _, p_val_g1 = fisher_exact([[2,1],[0,3]])
        self.assertAlmostEqual(gene1_res['PValue'], p_val_g1)
        self.assertTrue(gene1_res['RankMetric'] > 0) # Enriched in high

        # GENE2: High (0/3 mut), Low (2/3 mut)
        gene2_res = results_df[results_df['Gene'] == 'GENE2'].iloc[0]
        # table = [[0, 3], [2, 1]] -> OR = 0, p-val
        _, p_val_g2 = fisher_exact([[0,3],[2,1]])
        self.assertAlmostEqual(gene2_res['PValue'], p_val_g2)
        self.assertTrue(gene2_res['RankMetric'] < 0) # Enriched in low
        
        # GENE3: High (1/3 mut), Low (1/3 mut)
        gene3_res = results_df[results_df['Gene'] == 'GENE3'].iloc[0]
        # table = [[1,2],[1,2]] -> OR = 1, p-val = 1
        self.assertAlmostEqual(gene3_res['PValue'], 1.0)
        self.assertAlmostEqual(gene3_res['RankMetric'], 0.0) # np.log(1) = 0

        # GENE4: High (0/3 mut), Low (0/3 mut)
        gene4_res = results_df[results_df['Gene'] == 'GENE4'].iloc[0]
        # table = [[0,3],[0,3]] -> OR = nan (scipy), p-val = 1
        self.assertAlmostEqual(gene4_res['PValue'], 1.0)
        # pseudo_oddsratio will be (0.5*3.5)/(3.5*0.5) = 1. Rank metric = 0.
        self.assertAlmostEqual(gene4_res['RankMetric'], 0.0)


    def test_p_value_capping(self):
        # Test p-value capping (p-value = 0 becomes 1e-300)
        mutation_data = {'GENE_EXTREME': [1,1,1, 0,0,0]} # All high mutated, no low mutated
        samples = ['h1','h2','h3', 'l1','l2','l3']
        mutation_matrix = pd.DataFrame(mutation_data, index=samples)
        high_ids = {'h1','h2','h3'}
        low_ids = {'l1','l2','l3'}
        
        # This specific table [[3,0],[0,3]] gives p-value = 0.05 (for alternative='two-sided')
        # To get p-value of 0, we need a more extreme table that scipy might return as 0
        # However, direct mocking of fisher_exact is better to force p_value = 0
        with patch('generate_gsea_ranked_list.fisher_exact') as mock_fisher:
            mock_fisher.return_value = (1.0, 0.0) # OR=1, p_value=0
            results_df = perform_differential_analysis(mutation_matrix, high_ids, low_ids, {'GENE_EXTREME'})
            
        gene_res = results_df[results_df['Gene'] == 'GENE_EXTREME'].iloc[0]
        expected_rank_metric = -np.log10(1e-300) * np.sign(np.log((3+0.5)*(3+0.5) / ((0+0.5)*(0+0.5)))) # Sign of log(pseudo_OR)
        self.assertAlmostEqual(gene_res['RankMetric'], expected_rank_metric)
        self.assertEqual(gene_res['PValue'], 0.0) # Original p-value stored

    def test_perform_differential_analysis_no_genes(self):
        mutation_matrix = pd.DataFrame(index=['s1', 's2']) # No columns/genes
        high_ids = {'s1'}
        low_ids = {'s2'}
        results_df = perform_differential_analysis(mutation_matrix, high_ids, low_ids, set())
        self.assertTrue(results_df.empty)


    # --- Test for save_ranked_list ---
    @patch('pandas.DataFrame.to_csv')
    def test_save_ranked_list_successful(self, mock_to_csv):
        args = self._create_mock_args()
        results_data = {'Gene': ['GENE1', 'GENE2'], 'RankMetric': [10, -5], 'PValue': [0.01, 0.05], 'OddsRatio': [2, 0.5]}
        results_df = pd.DataFrame(results_data)

        save_ranked_list(results_df, args.output_ranked_gene_file)
        
        mock_to_csv.assert_called_once_with(
            args.output_ranked_gene_file, sep='\t', index=False, header=False
        )
    
    @patch("builtins.open", new_callable=mock_open)
    def test_save_ranked_list_empty_results(self, mock_file_open):
        args = self._create_mock_args()
        empty_results_df = pd.DataFrame(columns=['Gene', 'RankMetric', 'PValue', 'OddsRatio'])
        
        save_ranked_list(empty_results_df, args.output_ranked_gene_file)
        
        # Check if open was called to create an empty file
        mock_file_open.assert_called_once_with(args.output_ranked_gene_file, 'w')
        # pandas.to_csv should not be called if the DataFrame is empty by the logic in save_ranked_list
        # (The function itself creates an empty file if results_df is empty)

    @patch('pandas.DataFrame.to_csv', side_effect=IOError("Failed to write"))
    def test_save_ranked_list_failure(self, mock_to_csv):
        args = self._create_mock_args()
        results_data = {'Gene': ['GENE1'], 'RankMetric': [10]}
        results_df = pd.DataFrame(results_data)
        
        with self.assertRaises(ScriptLogicError, msg="Should raise ScriptLogicError on save failure"):
            save_ranked_list(results_df, args.output_ranked_gene_file)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
```
