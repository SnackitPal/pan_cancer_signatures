import unittest
from unittest.mock import patch, MagicMock, call, ANY
import argparse
import os
import sys
import io # For capturing stderr
import subprocess # For CompletedProcess

# Adjust sys.path to allow direct import of the script under test
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import main_pipeline # Import the module

class TestMainPipeline(unittest.TestCase):
    """Test suite for main_pipeline.py script."""

    def setUp(self):
        """Set up a base Namespace object for args for each test."""
        self.base_args = argparse.Namespace(
            cohort_list="TCGA-LUAD,TCGA-BRCA",
            k_lda=5,
            lda_seed=42,
            ref_genome_fasta="dummy_ref.fa", # Now required, added dummy
            cosmic_signatures_file="dummy_cosmic.tsv", # Now required, added dummy
            gmt_file="data/gene_sets/h.all.v2023.1.Hs.symbols.gmt",
            exome_size_mb=30.0,
            gsea_pairs="Signature_1:TCGA-LUAD,Signature_2:TCGA-BRCA",
            base_data_dir="./data_test/", 
            base_results_dir="./results_test/",
            report_output_file="./results_test/pdf_reports/Pipeline_Report_Test.pdf",
            run_stages="all",
            gsea_min_gene_set_size=15,
            gsea_max_gene_set_size=500,
            gsea_permutation_num=1000,
            gsea_plot_top_n=20,
            gsea_plot_fdr_threshold=0.25,
            regression_min_mutations=3,
            diffmut_min_group_size=10,
            diffmut_high_quantile=0.75,
            diffmut_low_quantile=0.25
        )

    def _get_minimal_required_args_list(self):
        """Helper to get a list of minimal required arguments with dummy values."""
        # Based on current create_parser in main_pipeline.py
        return [
            '--cohort_list', 'TCGA-TEST',
            '--ref_genome_fasta', 'dummy_ref.fa',
            '--cosmic_signatures_file', 'dummy_cosmic.tsv',
            '--gmt_file', 'dummy.gmt',
            '--gsea_pairs', 'Sig1:TCGA-TEST'
        ]
    
    def _get_all_args_with_specific_values_dict(self):
        return {
            '--cohort_list': 'TCGA-A,TCGA-B',
            '--k_lda': '7',
            '--lda_seed': '100',
            '--ref_genome_fasta': 'ref_custom.fa',
            '--cosmic_signatures_file': 'cosmic_custom.tsv',
            '--gmt_file': 'hallmark_custom.gmt',
            '--exome_size_mb': '33.3',
            '--gsea_pairs': 'S1:TCGA-A,S2:TCGA-B',
            '--base_data_dir': 'data_custom/',
            '--base_results_dir': 'results_custom/',
            '--report_output_file': 'custom_report.pdf',
            '--run_stages': 'calculate_tmb,generate_report'
            # Not including all GSEA/regression/diffmut specific params here for brevity,
            # as their defaults are tested below, and their presence in commands is tested elsewhere.
        }

    def test_argument_parser_creation(self):
        """Tests the argument parsing functionality of create_parser()."""
        parser = main_pipeline.create_parser()
        
        args_dict = self._get_all_args_with_specific_values_dict()
        args_list_all = [item for pair in args_dict.items() for item in pair]
        
        args = parser.parse_args(args_list_all)
        self.assertEqual(args.cohort_list, args_dict['--cohort_list'])
        self.assertEqual(args.k_lda, int(args_dict['--k_lda']))
        self.assertEqual(args.ref_genome_fasta, args_dict['--ref_genome_fasta']) # Now required
        self.assertEqual(args.cosmic_signatures_file, args_dict['--cosmic_signatures_file']) # Now required

        # Test Case: Default values
        required_args_for_defaults_test = self._get_minimal_required_args_list()
        args_defaults = parser.parse_args(required_args_for_defaults_test)
        self.assertEqual(args_defaults.run_stages, "all")
        self.assertEqual(args_defaults.k_lda, 5)

        # Test Case: Missing Required Arguments
        required_arg_names_to_test = ['--cohort_list', '--gmt_file', '--gsea_pairs', '--ref_genome_fasta', '--cosmic_signatures_file'] 
        base_args_for_missing_test = {
            '--cohort_list': 'C', '--gmt_file': 'G', '--gsea_pairs': 'P', 
            '--ref_genome_fasta': 'R.fa', '--cosmic_signatures_file': 'CS.tsv'
        }
        for arg_to_omit in required_arg_names_to_test:
            temp_args_list = []
            for k,v in base_args_for_missing_test.items():
                if k != arg_to_omit: temp_args_list.extend([k,v])
            # Need to ensure all truly required args are present except the one being tested for omission
            # This loop structure needs to be careful. A simpler way is to take a full valid list and remove one by one.
            full_valid_list = self._get_minimal_required_args_list()
            idx_to_omit = -1
            try: # Find index of arg_to_omit and its value to remove both
                idx_to_omit = full_valid_list.index(arg_to_omit)
                current_test_list = full_valid_list[:idx_to_omit] + full_valid_list[idx_to_omit+2:]
                with self.assertRaises(SystemExit, msg=f"Should exit when {arg_to_omit} is missing"):
                    parser.parse_args(current_test_list)
            except ValueError: # If arg_to_omit is not in full_valid_list (e.g. if it was a default before)
                pass # This test structure for missing args needs to be robust to which args are truly required vs having defaults.
                     # The loop as written before was fine for when all tested args were required.
                     # The create_parser defines which are required.

        with self.assertRaises(SystemExit, msg="Should exit when no arguments are provided"):
            parser.parse_args([])


    # --- Tests for _orchestrate_pipeline (main logic) ---
    # Common mocks for orchestration tests
    COMMON_PATCHES = {
        'os.makedirs': MagicMock(),
        'os.path.exists': MagicMock(return_value=True), # Assume scripts/dirs exist
        'sys.executable': 'python_mock_executable', # Mock via sys.executable
        'sys.exit': MagicMock() # To prevent test runner exit and check calls
    }

    @patch.multiple('main_pipeline', **COMMON_PATCHES)
    @patch('main_pipeline.subprocess.run')
    def test_command_construction_download_data(self, mock_subprocess_run, **kwargs):
        args = self.base_args; args.run_stages = "download_data"
        mock_subprocess_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout="OK")
        main_pipeline._orchestrate_pipeline(args)
        mock_subprocess_run.assert_called_once()
        cmd = mock_subprocess_run.call_args[0][0]
        self.assertEqual(cmd[0], 'python_mock_executable')
        self.assertTrue(cmd[1].endswith(os.path.join('scripts', 'download_tcga_mafs.py')))
        self.assertIn('--project_ids', cmd); self.assertIn(args.cohort_list, cmd)
        self.assertIn('--output_dir', cmd); self.assertIn(os.path.join(args.base_data_dir, "raw_mafs"), cmd)
        kwargs['sys.exit'].assert_not_called()

    @patch.multiple('main_pipeline', **COMMON_PATCHES)
    @patch('main_pipeline.subprocess.run')
    def test_command_construction_preprocess_data(self, mock_subprocess_run, **kwargs):
        args = self.base_args; args.run_stages = "preprocess_data"
        mock_subprocess_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout="OK")
        main_pipeline._orchestrate_pipeline(args)
        mock_subprocess_run.assert_called_once()
        cmd = mock_subprocess_run.call_args[0][0]
        self.assertTrue(cmd[1].endswith(os.path.join('scripts', 'preprocess_mafs.py')))
        self.assertIn('--maf_input_dir', cmd); self.assertIn(os.path.join(args.base_data_dir, "raw_mafs"), cmd)
        self.assertIn('--ref_genome_fasta', cmd); self.assertIn(args.ref_genome_fasta, cmd)
        self.assertIn('--output_matrix_file', cmd); self.assertIn(os.path.join(args.base_results_dir, "tables", "mutation_catalog.csv"), cmd)
        kwargs['sys.exit'].assert_not_called()

    @patch.multiple('main_pipeline', **COMMON_PATCHES)
    @patch('main_pipeline.subprocess.run')
    def test_command_construction_train_lda_model(self, mock_subprocess_run, **kwargs):
        args = self.base_args; args.run_stages = "train_lda_model"
        mock_subprocess_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout="OK")
        main_pipeline._orchestrate_pipeline(args)
        mock_subprocess_run.assert_called_once()
        cmd = mock_subprocess_run.call_args[0][0]
        self.assertTrue(cmd[1].endswith(os.path.join('scripts', 'train_lda_model.py')))
        self.assertIn('--matrix_path', cmd); self.assertIn(os.path.join(args.base_results_dir, "tables", "mutation_catalog.csv"), cmd)
        self.assertIn('--output_dir_models', cmd); self.assertIn(os.path.join(args.base_results_dir, "lda_models"), cmd) # Script expects output_dir_models
        self.assertIn('--num_signatures', cmd); self.assertIn(str(args.k_lda), cmd)
        self.assertIn('--random_seed', cmd); self.assertIn(str(args.lda_seed), cmd)
        kwargs['sys.exit'].assert_not_called()

    @patch.multiple('main_pipeline', **COMMON_PATCHES)
    @patch('main_pipeline.subprocess.run')
    def test_command_construction_visualize_signatures(self, mock_subprocess_run, **kwargs):
        args = self.base_args; args.run_stages = "visualize_signatures"
        mock_subprocess_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout="OK")
        main_pipeline._orchestrate_pipeline(args)
        mock_subprocess_run.assert_called_once()
        cmd = mock_subprocess_run.call_args[0][0]
        self.assertTrue(cmd[1].endswith(os.path.join('scripts', 'visualize_signatures.py')))
        self.assertIn('--signature_profiles_file', cmd)
        self.assertIn(os.path.join(args.base_results_dir, "lda_models", f"signature_profiles_k{args.k_lda}_seed{args.lda_seed}.csv"), cmd)
        self.assertIn('--output_dir_figures', cmd); self.assertIn(os.path.join(args.base_results_dir, "figures", "signatures"), cmd)
        kwargs['sys.exit'].assert_not_called()

    @patch.multiple('main_pipeline', **COMMON_PATCHES)
    @patch('main_pipeline.subprocess.run')
    def test_command_construction_compare_to_cosmic(self, mock_subprocess_run, **kwargs):
        args = self.base_args; args.run_stages = "compare_to_cosmic"
        mock_subprocess_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout="OK")
        main_pipeline._orchestrate_pipeline(args)
        mock_subprocess_run.assert_called_once()
        cmd = mock_subprocess_run.call_args[0][0]
        self.assertTrue(cmd[1].endswith(os.path.join('scripts', 'compare_to_cosmic.py')))
        self.assertIn('--discovered_profiles_file', cmd)
        self.assertIn(os.path.join(args.base_results_dir, "lda_models", f"signature_profiles_k{args.k_lda}_seed{args.lda_seed}.csv"), cmd)
        self.assertIn('--cosmic_profiles_file', cmd); self.assertIn(args.cosmic_signatures_file, cmd)
        self.assertIn('--output_dir_comparison', cmd); self.assertIn(os.path.join(args.base_results_dir, "comparison"), cmd)
        kwargs['sys.exit'].assert_not_called()

    @patch.multiple('main_pipeline', **COMMON_PATCHES)
    @patch('main_pipeline.subprocess.run')
    def test_command_construction_generate_sample_map(self, mock_subprocess_run, **kwargs):
        args = self.base_args; args.run_stages = "generate_sample_map"
        mock_subprocess_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout="OK")
        main_pipeline._orchestrate_pipeline(args)
        mock_subprocess_run.assert_called_once()
        cmd = mock_subprocess_run.call_args[0][0]
        self.assertTrue(cmd[1].endswith(os.path.join('scripts', 'generate_sample_cohort_map.py')))
        self.assertIn('--maf_input_dir', cmd); self.assertIn(os.path.join(args.base_data_dir, "raw_mafs"), cmd)
        self.assertIn('--output_map_file', cmd); self.assertIn(os.path.join(args.base_data_dir, "processed", "sample_cohort_map.csv"), cmd)
        kwargs['sys.exit'].assert_not_called()

    @patch.multiple('main_pipeline', **COMMON_PATCHES)
    @patch('main_pipeline.subprocess.run')
    def test_command_construction_analyze_exposures(self, mock_subprocess_run, **kwargs):
        args = self.base_args; args.run_stages = "analyze_exposures"
        mock_subprocess_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout="OK")
        main_pipeline._orchestrate_pipeline(args)
        mock_subprocess_run.assert_called_once()
        cmd = mock_subprocess_run.call_args[0][0]
        self.assertTrue(cmd[1].endswith(os.path.join('scripts', 'analyze_patient_exposures.py')))
        self.assertIn('--exposures_file', cmd)
        self.assertIn(os.path.join(args.base_results_dir, "lda_models", f"patient_exposures_k{args.k_lda}_seed{args.lda_seed}.csv"), cmd)
        self.assertIn('--sample_map_file', cmd)
        self.assertIn(os.path.join(args.base_data_dir, "processed", "sample_cohort_map.csv"), cmd)
        self.assertIn('--output_dir_figures', cmd); self.assertIn(os.path.join(args.base_results_dir, "figures", "exposures"), cmd)
        kwargs['sys.exit'].assert_not_called()

    @patch.multiple('main_pipeline', **COMMON_PATCHES)
    @patch('main_pipeline.subprocess.run')
    def test_command_construction_calculate_tmb_updated_paths(self, mock_subprocess_run, **kwargs):
        args = self.base_args; args.run_stages = "calculate_tmb"
        mock_subprocess_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout="OK")
        main_pipeline._orchestrate_pipeline(args)
        mock_subprocess_run.assert_called_once()
        cmd = mock_subprocess_run.call_args[0][0]
        self.assertIn(os.path.join(args.base_data_dir, "raw_mafs"), cmd) # Check for raw_mafs
        kwargs['sys.exit'].assert_not_called()

    @patch.multiple('main_pipeline', **COMMON_PATCHES)
    @patch('main_pipeline.subprocess.run')
    def test_pipeline_halts_on_script_failure_updated(self, mock_subprocess_run, **kwargs):
        args = self.base_args; args.run_stages = "calculate_tmb,generate_report"
        mock_subprocess_run.return_value = subprocess.CompletedProcess(args=[], returncode=1, stderr="Failure")
        main_pipeline._orchestrate_pipeline(args)
        mock_subprocess_run.assert_called_once() 
        kwargs['sys.exit'].assert_called_once_with(1)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
