import unittest
from unittest import mock
from unittest.mock import MagicMock, patch, mock_open, call # Added call
import pandas as pd
import numpy as np
import argparse
import sys
import os
import io # For StringIO

# Add scripts directory to sys.path to allow direct import
scripts_dir = os.path.join(os.path.dirname(__file__), '..', 'scripts')
if scripts_dir not in sys.path: # Ensure it's not added multiple times
    sys.path.insert(0, os.path.abspath(scripts_dir))

import scripts.compare_to_cosmic as compare_to_cosmic # Script to be tested

class TestCompareToCosmic(unittest.TestCase):

    def setUp(self):
        """Set up for test methods."""
        self.contexts = [f'ctx{i}' for i in range(96)]
        
        self.sample_discovered_df = pd.DataFrame(
            np.random.rand(3, 96), 
            index=['Sig1', 'Sig2', 'Sig3'], 
            columns=self.contexts
        )
        
        self.sample_cosmic_df_pre_transpose = pd.DataFrame(
            np.random.rand(96, 5), 
            index=self.contexts, 
            columns=[f'SBS{i}' for i in range(1, 6)]
        )
        # This is what it looks like after transpose in the script
        self.sample_cosmic_df_transposed = self.sample_cosmic_df_pre_transpose.T.copy()
        # Ensure it sums to 1 for testing normalization logic bypass
        self.sample_cosmic_df_transposed_normalized = self.sample_cosmic_df_transposed.apply(lambda x: x / x.sum(), axis=1)


        # Mock sys.stdout to capture print statements
        # self.patcher_stdout will be started in each test method where needed
        self.patcher_stdout = patch('sys.stdout', new_callable=io.StringIO)
        
        # Suppress print statements from the script itself (redundant if stdout is captured, but good for isolation)
        self.patcher_print = patch('builtins.print') 
        self.mock_bprint = self.patcher_print.start() # Start it here, stop in tearDown

        self.default_args = argparse.Namespace(
            discovered_profiles_path='dummy_discovered.csv',
            cosmic_profiles_path='dummy_cosmic.tsv',
            output_csv_path='dummy_output.csv',
            output_heatmap_path=None,
            top_n_matches=3
        )

    def tearDown(self):
        self.patcher_print.stop() # Stop builtins.print patcher

    @patch('argparse.ArgumentParser.parse_args')
    @patch('pandas.read_csv', side_effect=FileNotFoundError) # Mock to prevent file ops
    @patch('sys.exit') # Mock sys.exit to prevent test runner from exiting
    def test_argument_parsing(self, mock_sys_exit, mock_read_csv, mock_parse_args):
        """Test argument parsing logic by checking if main uses the parsed args."""
        # Prepare a specific args namespace for this test
        test_args_namespace = argparse.Namespace(
            discovered_profiles_path='path1.csv',
            cosmic_profiles_path='path2.tsv',
            output_csv_path='out.csv',
            output_heatmap_path='heatmap.png',
            top_n_matches=5
        )
        mock_parse_args.return_value = test_args_namespace

        # sys.argv is not directly used if parse_args is mocked, but good practice for completeness
        with patch('sys.argv', ['compare_to_cosmic.py', '--discovered_profiles_path', 'path1.csv', 
                                '--cosmic_profiles_path', 'path2.tsv', '--output_csv_path', 'out.csv',
                                '--output_heatmap_path', 'heatmap.png', '--top_n_matches', '5']):
            compare_to_cosmic.main()

        mock_parse_args.assert_called_once()
        # Check that pandas.read_csv was called with the path from the mocked args
        # It will be called twice, first for discovered, then for cosmic
        self.assertGreaterEqual(mock_read_csv.call_count, 1)
        # The actual file paths are from test_args_namespace
        mock_read_csv.assert_any_call('path1.csv', index_col=0)
        # If it gets past the first read_csv error (FileNotFoundError), it would call for cosmic
        # The side_effect=FileNotFoundError means it will error out after the first call.
        # This is sufficient to test that discovered_profiles_path was used.

    @patch('pandas.read_csv')
    @patch('sys.exit') # Mock to prevent exits during tests
    @patch('scripts.compare_to_cosmic.cosine_similarity', return_value=np.array([[1.0]])) # Mock downstream
    @patch('pandas.DataFrame.to_csv') # Mock downstream
    def test_data_loading_and_preparation(self, mock_df_to_csv, mock_cosine_sim, mock_sys_exit, mock_pd_read_csv):
        """Test data loading, preparation, and validation (transpose, normalization)."""
        mock_stdout = self.patcher_stdout.start() # Start stdout capture

        # Case 1: COSMIC needs normalization
        cosmic_needs_norm = self.sample_cosmic_df_pre_transpose.copy() # contexts x sigs
        # Modify so it doesn't sum to 1
        cosmic_needs_norm.iloc[:,0] = cosmic_needs_norm.iloc[:,0] * 2 
        
        mock_pd_read_csv.side_effect = [
            self.sample_discovered_df.copy(), 
            cosmic_needs_norm 
        ]
        
        with patch('argparse.ArgumentParser.parse_args', return_value=self.default_args):
            compare_to_cosmic.main()

        self.assertEqual(mock_pd_read_csv.call_count, 2)
        mock_pd_read_csv.assert_any_call(self.default_args.discovered_profiles_path, index_col=0)
        mock_pd_read_csv.assert_any_call(self.default_args.cosmic_profiles_path, sep='\t', index_col=0)
        
        output = mock_stdout.getvalue()
        self.assertIn("Normalizing COSMIC signature profiles", output)
        self.assertIn("COSMIC profiles normalized successfully.", output)

        # Case 2: COSMIC already normalized
        mock_pd_read_csv.reset_mock()
        mock_stdout.truncate(0)
        mock_stdout.seek(0)

        # Use the pre-transposed and pre-normalized version for the side_effect
        # The script expects contexts x sigs from read_csv for COSMIC, then transposes
        # So, we need to provide sample_cosmic_df_pre_transpose where columns (signatures) sum to 1
        # This is tricky. Let's provide a version that, *after transpose*, sums to 1.
        cosmic_already_norm_pre_T = self.sample_cosmic_df_transposed_normalized.T.copy()

        mock_pd_read_csv.side_effect = [
            self.sample_discovered_df.copy(),
            cosmic_already_norm_pre_T
        ]
        with patch('argparse.ArgumentParser.parse_args', return_value=self.default_args):
            compare_to_cosmic.main()
        
        output = mock_stdout.getvalue()
        self.assertIn("COSMIC profiles appear to be already normalized", output)
        
        self.patcher_stdout.stop() # Stop stdout capture

    @patch('scripts.compare_to_cosmic.cosine_similarity')
    @patch('pandas.DataFrame.to_csv') # Mock saving
    @patch('sys.exit')
    def test_context_alignment(self, mock_sys_exit, mock_to_csv, mock_cosine_similarity):
        """Test context alignment logic, including warnings and errors."""
        mock_stdout = self.patcher_stdout.start()

        # Case 1: Common contexts are < 96 (but > 10)
        discovered_less_contexts = self.sample_discovered_df.iloc[:, :90].copy() # 90 contexts
        cosmic_less_contexts = self.sample_cosmic_df_transposed_normalized.iloc[:, :90].copy() # 90 contexts
        
        with patch('pandas.read_csv', side_effect=[discovered_less_contexts, cosmic_less_contexts.T.copy()]), \
             patch('argparse.ArgumentParser.parse_args', return_value=self.default_args):
            compare_to_cosmic.main()
        
        output = mock_stdout.getvalue()
        self.assertIn("Warning: Number of common contexts (90) is less than", output)
        # Check that cosine_similarity was called with data of shape (x, 90)
        self.assertEqual(mock_cosine_similarity.call_args[0][0].shape[1], 90)
        self.assertEqual(mock_cosine_similarity.call_args[0][1].shape[1], 90)


        # Case 2: Too few common contexts (e.g., 5)
        mock_stdout.truncate(0); mock_stdout.seek(0)
        mock_cosine_similarity.reset_mock()
        mock_sys_exit.reset_mock()
        discovered_few_contexts = self.sample_discovered_df.iloc[:, :5].copy() # 5 contexts
        cosmic_few_contexts = self.sample_cosmic_df_transposed_normalized.iloc[:, :5].copy()
        
        with patch('pandas.read_csv', side_effect=[discovered_few_contexts, cosmic_few_contexts.T.copy()]), \
             patch('argparse.ArgumentParser.parse_args', return_value=self.default_args):
            compare_to_cosmic.main()
        
        output = mock_stdout.getvalue()
        self.assertIn("Error: Only 5 common contexts found.", output)
        # mock_sys_exit.assert_called_once() # Script uses 'return'

        # Case 3: No common contexts
        mock_stdout.truncate(0); mock_stdout.seek(0)
        mock_cosine_similarity.reset_mock()
        mock_sys_exit.reset_mock()
        # Create DFs with completely different column names
        discovered_no_common = self.sample_discovered_df.copy()
        discovered_no_common.columns = [f'd_ctx{i}' for i in range(96)]
        cosmic_no_common = self.sample_cosmic_df_transposed_normalized.copy()
        cosmic_no_common.columns = [f'c_ctx{i}' for i in range(96)]

        with patch('pandas.read_csv', side_effect=[discovered_no_common, cosmic_no_common.T.copy()]), \
             patch('argparse.ArgumentParser.parse_args', return_value=self.default_args):
            compare_to_cosmic.main()
            
        output = mock_stdout.getvalue()
        self.assertIn("Error: No common contexts found", output)
        # mock_sys_exit.assert_called_once() # Script uses 'return'

        self.patcher_stdout.stop()

    @patch('pandas.read_csv')
    @patch('sklearn.metrics.pairwise.cosine_similarity')
    @patch('pandas.DataFrame.to_csv')
    @patch('os.makedirs') # To check if output CSV dir is created
    def test_cosine_similarity_calculation_and_save(self, mock_os_makedirs, mock_to_csv, 
                                                    mock_cosine_similarity, mock_read_csv):
        """Test cosine similarity calculation and saving the matrix."""
        # Provide aligned DFs for this test
        aligned_discovered_df = self.sample_discovered_df.copy()
        aligned_cosmic_df = self.sample_cosmic_df_transposed_normalized.copy()
        # Ensure they share the exact same columns for this specific test part
        aligned_cosmic_df = aligned_cosmic_df[aligned_discovered_df.columns]

        mock_read_csv.side_effect = [aligned_discovered_df, aligned_cosmic_df.T.copy()]
        
        # Define a sample similarity matrix to be returned by the mock
        expected_similarity_array = np.array([[0.9, 0.1, 0.2], [0.05, 0.95, 0.3], [0.15, 0.25, 0.85]])
        # Adjust shape to match sample_discovered_df (3 sigs) and sample_cosmic_df (5 sigs)
        expected_similarity_array = np.random.rand(3, 5) 
        mock_cosine_similarity.return_value = expected_similarity_array

        with patch('argparse.ArgumentParser.parse_args', return_value=self.default_args):
            compare_to_cosmic.main()

        mock_cosine_similarity.assert_called_once()
        # Check if the arrays passed to cosine_similarity match the .values of the DFs
        np.testing.assert_array_equal(mock_cosine_similarity.call_args[0][0], aligned_discovered_df.values)
        np.testing.assert_array_equal(mock_cosine_similarity.call_args[0][1], aligned_cosmic_df.values)

        mock_to_csv.assert_called_once_with(self.default_args.output_csv_path)
        # Check the DataFrame that was passed to to_csv
        df_passed_to_csv = mock_to_csv.call_args[0][0] # The DataFrame instance is the first arg
        self.assertIsInstance(df_passed_to_csv, pd.DataFrame)
        self.assertEqual(df_passed_to_csv.shape, expected_similarity_array.shape)
        self.assertListEqual(df_passed_to_csv.index.tolist(), aligned_discovered_df.index.tolist())
        self.assertListEqual(df_passed_to_csv.columns.tolist(), aligned_cosmic_df.index.tolist())
        np.testing.assert_array_almost_equal(df_passed_to_csv.values, expected_similarity_array)


    @patch('pandas.read_csv')
    @patch('sklearn.metrics.pairwise.cosine_similarity')
    @patch('pandas.DataFrame.to_csv') # Mock to prevent actual saving
    def test_top_n_matches_reporting(self, mock_to_csv, mock_cosine_similarity, mock_read_csv):
        """Test reporting of top N matches."""
        mock_stdout = self.patcher_stdout.start()

        # Create a specific similarity matrix for predictable top N results
        # Discovered: SigA, SigB. COSMIC: SBS1, SBS2, SBS3
        similarity_data = {
            'SBS1': [0.95, 0.10],
            'SBS2': [0.20, 0.92],
            'SBS3': [0.50, 0.40]
        }
        sample_similarity_df = pd.DataFrame(similarity_data, index=['SigA', 'SigB'])
        
        # Mock cosine_similarity to return this matrix's values
        mock_cosine_similarity.return_value = sample_similarity_df.values
        
        # Mock read_csv to provide DFs that would result in this similarity_df structure
        # Discovered Df: index SigA, SigB
        discovered_df_for_topn = pd.DataFrame(index=['SigA', 'SigB'], columns=self.contexts)
        # Cosmic Df: index SBS1, SBS2, SBS3 (after transpose)
        cosmic_df_for_topn = pd.DataFrame(index=['SBS1', 'SBS2', 'SBS3'], columns=self.contexts)
        
        mock_read_csv.side_effect = [discovered_df_for_topn, cosmic_df_for_topn.T.copy()]

        # Args for this test
        args_top_n = argparse.Namespace(
            discovered_profiles_path='d.csv', cosmic_profiles_path='c.tsv',
            output_csv_path='o.csv', output_heatmap_path=None, top_n_matches=2
        )
        with patch('argparse.ArgumentParser.parse_args', return_value=args_top_n):
            compare_to_cosmic.main()

        output = mock_stdout.getvalue()
        self.assertIn("Top 2 COSMIC Matches", output)
        self.assertIn("Discovered SigA: SBS1 (Similarity: 0.950), SBS3 (Similarity: 0.500)", output)
        self.assertIn("Discovered SigB: SBS2 (Similarity: 0.920), SBS3 (Similarity: 0.400)", output)
        
        self.patcher_stdout.stop()

    @patch('pandas.read_csv')
    @patch('sklearn.metrics.pairwise.cosine_similarity')
    @patch('pandas.DataFrame.to_csv') # Mock to prevent actual saving
    @patch('seaborn.heatmap')
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    @patch('os.makedirs')
    @patch('matplotlib.pyplot.figure') # To mock plt.figure call
    def test_heatmap_generation(self, mock_plt_figure, mock_os_makedirs, mock_close, mock_savefig, 
                                mock_heatmap, mock_to_csv, mock_cosine_similarity, mock_read_csv):
        """Test heatmap generation logic (path provided and not provided)."""
        mock_stdout = self.patcher_stdout.start()

        # Setup: Ensure cosine_similarity returns a valid matrix
        sample_similarity_array = np.random.rand(3, 5) # 3 discovered, 5 COSMIC
        mock_cosine_similarity.return_value = sample_similarity_array
        # Ensure read_csv provides DFs that match these dimensions
        mock_read_csv.side_effect = [
            self.sample_discovered_df, # 3 discovered sigs
            self.sample_cosmic_df_pre_transpose # 5 COSMIC sigs
        ]

        # Case 1: Heatmap path provided
        args_with_heatmap = argparse.Namespace(
            discovered_profiles_path='d.csv', cosmic_profiles_path='c.tsv',
            output_csv_path='o.csv', output_heatmap_path='heatmap.png', top_n_matches=3
        )
        with patch('argparse.ArgumentParser.parse_args', return_value=args_with_heatmap):
            compare_to_cosmic.main()

        mock_os_makedirs.assert_any_call(os.path.dirname('heatmap.png'), exist_ok=True) # Check dir creation
        mock_heatmap.assert_called_once()
        # Check some key args of heatmap call
        self.assertIsInstance(mock_heatmap.call_args[0][0], pd.DataFrame) # Data arg
        self.assertEqual(mock_heatmap.call_args[0][0].shape, sample_similarity_array.shape)
        self.assertTrue(mock_heatmap.call_args[1]['annot']) # annot=True by default if not too large
        
        mock_savefig.assert_called_once_with('heatmap.png', dpi=300, bbox_inches='tight')
        mock_close.assert_called_once()
        output = mock_stdout.getvalue()
        self.assertIn("Generating heatmap and saving to: heatmap.png", output)

        # Case 2: Heatmap path not provided
        mock_stdout.truncate(0); mock_stdout.seek(0)
        mock_os_makedirs.reset_mock()
        mock_heatmap.reset_mock()
        mock_savefig.reset_mock()
        mock_close.reset_mock()
        # Reset read_csv side effect as it's consumed
        mock_read_csv.side_effect = [
            self.sample_discovered_df, 
            self.sample_cosmic_df_pre_transpose
        ]

        args_no_heatmap = argparse.Namespace(
            discovered_profiles_path='d.csv', cosmic_profiles_path='c.tsv',
            output_csv_path='o.csv', output_heatmap_path=None, top_n_matches=3
        )
        with patch('argparse.ArgumentParser.parse_args', return_value=args_no_heatmap):
            compare_to_cosmic.main()

        self.assertFalse(mock_heatmap.called)
        self.assertFalse(mock_savefig.called)
        output = mock_stdout.getvalue()
        self.assertIn("Heatmap generation skipped", output)
        
        self.patcher_stdout.stop()

if __name__ == '__main__':
    unittest.main()
