import unittest
from unittest.mock import patch, MagicMock, mock_open
import argparse
import pandas as pd
import numpy as np
import sys
import os
import tempfile # For creating temporary files/directories

# Adjust sys.path to allow direct import of the script under test
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts')))

# Import functions from the script to be tested
from run_gsea_analysis import create_parser, run_gsea, main as run_gsea_main 

class TestRunGseaAnalysis(unittest.TestCase):
    """Test suite for run_gsea_analysis.py script."""

    def setUp(self):
        """Set up temporary files and directories for tests."""
        self.test_dir = tempfile.TemporaryDirectory()
        
        # Create a dummy ranked gene list file
        self.ranked_file = tempfile.NamedTemporaryFile(mode='w+', delete=False, dir=self.test_dir.name, suffix=".rnk")
        self.ranked_file.writelines(["GENE1\t10.0\n", "GENE2\t-5.0\n", "GENE3\t1.0\n"])
        self.ranked_file.close() # Close it so the script can open it

        # Create a dummy GMT file
        self.gmt_file = tempfile.NamedTemporaryFile(mode='w+', delete=False, dir=self.test_dir.name, suffix=".gmt")
        self.gmt_file.writelines(["SET1\tdesc1\tGENE1\tGENE3\n", "SET2\tdesc2\tGENE2\tGENE4\n"])
        self.gmt_file.close()

        # Define mock arguments
        self.mock_args = argparse.Namespace(
            ranked_gene_list_file=self.ranked_file.name,
            gene_sets_gmt_file=self.gmt_file.name,
            output_dir_gsea=os.path.join(self.test_dir.name, "gsea_output"),
            min_gene_set_size=5, # Adjusted for small test gene sets
            max_gene_set_size=500,
            permutation_num=100, # Reduced for faster tests if not mocking gseapy
            seed=42
        )

    def tearDown(self):
        """Clean up temporary files and directories."""
        os.unlink(self.ranked_file.name)
        os.unlink(self.gmt_file.name)
        self.test_dir.cleanup()

    def test_argument_parser(self):
        """Tests the argument parsing functionality (already verified, kept for completeness)."""
        parser = create_parser()
        args_list_full = [
            "--ranked_gene_list_file", "test.rnk", "--gene_sets_gmt_file", "test.gmt",
            "--output_dir_gsea", "test_output_gsea", "--min_gene_set_size", "20",
            "--max_gene_set_size", "400", "--permutation_num", "500", "--seed", "123"
        ]
        args = parser.parse_args(args_list_full)
        self.assertEqual(args.ranked_gene_list_file, "test.rnk")
        # ... (other assertions as in previous implementation)

    # --- Tests for file loading and directory creation ---
    def test_load_ranked_gene_list_successful(self):
        """Tests successful loading of a correctly formatted ranked gene list."""
        # run_gsea itself handles loading. We call it and check intermediate state or final outcome.
        # For this specific test, we'll patch gseapy.prerank to avoid running actual GSEA.
        with patch('run_gsea_analysis.gseapy.prerank', return_value=MagicMock(results=pd.DataFrame({'nes': [1]}))) as mock_gsea, \
             patch('run_gsea_analysis.os.makedirs') as mock_makedirs:
            try:
                run_gsea(self.mock_args) 
            except Exception as e:
                # If run_gsea raises an exception that's not FileNotFoundError or ValueError related to loading,
                # this test might fail, indicating an issue in the test setup or run_gsea logic beyond loading.
                self.fail(f"run_gsea raised an unexpected exception during successful load test: {e}")
            
            # Check that makedirs was called
            mock_makedirs.assert_called_once_with(self.mock_args.output_dir_gsea, exist_ok=True)
            # Check that gseapy.prerank was called, implying successful load
            mock_gsea.assert_called_once()
            # The first argument to prerank is 'rnk', which should be a pandas Series.
            call_args, _ = mock_gsea.call_args
            loaded_series = call_args[0] # or call_args.kwargs['rnk'] depending on how it's called
            self.assertIsInstance(loaded_series, pd.Series)
            self.assertEqual(len(loaded_series), 3)
            self.assertEqual(loaded_series['GENE1'], 10.0)


    def test_load_ranked_gene_list_file_not_found(self):
        """Tests behavior when the ranked gene list file is not found."""
        args_bad_file = self.mock_args_bad_file = argparse.Namespace(
            **vars(self.mock_args) # copy existing args
        )
        args_bad_file.ranked_gene_list_file = "non_existent_file.rnk"
        
        with self.assertRaises(FileNotFoundError):
            run_gsea(args_bad_file)

    def test_load_ranked_gene_list_incorrect_format(self):
        """Tests behavior with incorrectly formatted ranked gene list files."""
        # Case 1: Incorrect number of columns
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, dir=self.test_dir.name) as tmp_bad_format_1:
            tmp_bad_format_1.write("GENE1\t10.0\tEXTRA_COL\n")
            tmp_bad_format_1_name = tmp_bad_format_1.name
        
        args_bad_format_1 = argparse.Namespace(**vars(self.mock_args))
        args_bad_format_1.ranked_gene_list_file = tmp_bad_format_1_name
        with self.assertRaises(ValueError, msg="Should raise ValueError for incorrect column count"):
            run_gsea(args_bad_format_1)
        os.unlink(tmp_bad_format_1_name)

        # Case 2: Non-numeric rank metric
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, dir=self.test_dir.name) as tmp_bad_format_2:
            tmp_bad_format_2.write("GENE1\tNOT_A_NUMBER\n")
            tmp_bad_format_2_name = tmp_bad_format_2.name

        args_bad_format_2 = argparse.Namespace(**vars(self.mock_args))
        args_bad_format_2.ranked_gene_list_file = tmp_bad_format_2_name
        with self.assertRaises(ValueError, msg="Should raise ValueError for non-numeric rank metric"):
            run_gsea(args_bad_format_2)
        os.unlink(tmp_bad_format_2_name)

        # Case 3: Missing values
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, dir=self.test_dir.name) as tmp_bad_format_3:
            tmp_bad_format_3.write("GENE1\t\n") # Missing rank
            tmp_bad_format_3_name = tmp_bad_format_3.name
        
        args_bad_format_3 = argparse.Namespace(**vars(self.mock_args))
        args_bad_format_3.ranked_gene_list_file = tmp_bad_format_3_name
        with self.assertRaises(ValueError, msg="Should raise ValueError for missing values"):
            run_gsea(args_bad_format_3)
        os.unlink(tmp_bad_format_3_name)


    @patch('run_gsea_analysis.os.makedirs')
    def test_create_output_directory(self, mock_makedirs):
        """Tests that os.makedirs is called correctly to create the output directory."""
        # We only need to check if os.makedirs is called correctly.
        # Patch gseapy.prerank as well to prevent it from actually running.
        with patch('run_gsea_analysis.gseapy.prerank', return_value=MagicMock(results=pd.DataFrame({'nes': [1]}))):
            run_gsea(self.mock_args)
        mock_makedirs.assert_called_once_with(self.mock_args.output_dir_gsea, exist_ok=True)

    # --- Tests for gseapy.prerank call ---
    @patch('run_gsea_analysis.gseapy.prerank')
    @patch('run_gsea_analysis.os.makedirs') # Also mock makedirs
    def test_gseapy_prerank_successful_call(self, mock_makedirs, mock_gsea_prerank):
        """Tests a successful call to gseapy.prerank with correct parameters."""
        mock_gsea_prerank.return_value = MagicMock(results=pd.DataFrame({'nes': [1.0], 'fdr': [0.01], 'Term': ['SET1']}))
        
        run_gsea(self.mock_args)

        # Verify gseapy.prerank was called once
        mock_gsea_prerank.assert_called_once()
        
        # Verify call arguments
        call_args = mock_gsea_prerank.call_args[1] # Get kwargs
        self.assertIsInstance(call_args['rnk'], pd.Series)
        self.assertEqual(call_args['gene_sets'], self.gmt_file.name)
        self.assertEqual(call_args['outdir'], self.mock_args.output_dir_gsea)
        self.assertEqual(call_args['min_size'], self.mock_args.min_gene_set_size)
        self.assertEqual(call_args['max_size'], self.mock_args.max_gene_set_size)
        self.assertEqual(call_args['permutation_num'], self.mock_args.permutation_num)
        self.assertEqual(call_args['seed'], self.mock_args.seed)
        self.assertTrue(call_args['verbose'])
        # Success message is printed by run_gsea, could patch print to check if needed

    @patch('run_gsea_analysis.gseapy.prerank')
    @patch('run_gsea_analysis.os.makedirs')
    def test_gseapy_prerank_failure_empty_results(self, mock_makedirs, mock_gsea_prerank):
        """Tests GSEApy call that 'succeeds' but returns empty results."""
        mock_gsea_prerank.return_value = MagicMock(results=pd.DataFrame()) # Empty results
        
        with self.assertRaises(Exception, msg="Should raise Exception if GSEApy returns empty results"):
            run_gsea(self.mock_args)

    @patch('run_gsea_analysis.gseapy.prerank')
    @patch('run_gsea_analysis.os.makedirs')
    def test_gseapy_prerank_raises_exception(self, mock_makedirs, mock_gsea_prerank):
        """Tests handling of exceptions raised by gseapy.prerank."""
        mock_gsea_prerank.side_effect = Exception("GSEApy internal error")
        
        with self.assertRaises(Exception, msg="Should re-raise GSEApy's exception"):
            run_gsea(self.mock_args)
            
    @patch('run_gsea_analysis.gseapy.prerank') # Should not be called
    @patch('run_gsea_analysis.os.makedirs')
    def test_gmt_file_not_found(self, mock_makedirs, mock_gsea_prerank):
        """Tests error handling when the GMT gene sets file is not found."""
        args_bad_gmt = argparse.Namespace(**vars(self.mock_args))
        args_bad_gmt.gene_sets_gmt_file = "non_existent_gmt.gmt"
        
        with self.assertRaises(FileNotFoundError):
            run_gsea(args_bad_gmt)
        mock_gsea_prerank.assert_not_called() # Ensure GSEA was not even attempted

    # Test main function's error handling for completeness (optional, as individual functions are tested)
    @patch('run_gsea_analysis.run_gsea', side_effect=FileNotFoundError("Mocked FileNotFoundError"))
    @patch('sys.exit') # To prevent test runner from exiting
    def test_main_catches_file_not_found(self, mock_sys_exit, mock_run_gsea_func):
        run_gsea_main([
            "--ranked_gene_list_file", "dummy.rnk", 
            "--gene_sets_gmt_file", "dummy.gmt", 
            "--output_dir_gsea", "dummy_out"
        ])
        mock_sys_exit.assert_called_once_with(1)

    @patch('run_gsea_analysis.run_gsea', side_effect=ValueError("Mocked ValueError"))
    @patch('sys.exit')
    def test_main_catches_value_error(self, mock_sys_exit, mock_run_gsea_func):
        run_gsea_main(["--ranked_gene_list_file", "dummy.rnk", "--gene_sets_gmt_file", "dummy.gmt", "--output_dir_gsea", "dummy_out"])
        mock_sys_exit.assert_called_once_with(1)

    @patch('run_gsea_analysis.run_gsea', side_effect=Exception("Mocked Generic Exception"))
    @patch('sys.exit')
    def test_main_catches_generic_exception(self, mock_sys_exit, mock_run_gsea_func):
        run_gsea_main(["--ranked_gene_list_file", "dummy.rnk", "--gene_sets_gmt_file", "dummy.gmt", "--output_dir_gsea", "dummy_out"])
        mock_sys_exit.assert_called_once_with(1)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
