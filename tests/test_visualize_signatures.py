import unittest
from unittest import mock 
from unittest.mock import MagicMock, patch, call # Added call for checking multiple calls
import pandas as pd
import numpy as np
import argparse
import os
import sys

# Add scripts directory to sys.path to allow direct import
scripts_dir = os.path.join(os.path.dirname(__file__), '..', 'scripts')
if scripts_dir not in sys.path: # Ensure it's not added multiple times if tests are run together
    sys.path.insert(0, os.path.abspath(scripts_dir))

import visualize_signatures # Script to be tested
from signature_plotting_utils import STANDARD_96_CONTEXTS, MUTATION_TYPE_COLORS, MUTATION_TYPES

class TestVisualizeSignatures(unittest.TestCase):

    def setUp(self):
        """Set up for test methods."""
        # Suppress print statements from the script
        self.patcher_print = patch('builtins.print')
        self.mock_print = self.patcher_print.start()

        # Sample signature profiles DataFrame
        self.sample_num_signatures = 2
        self.sample_num_contexts = 96
        
        # Create contexts in a non-standard order initially for testing reordering
        shuffled_contexts = STANDARD_96_CONTEXTS[:]
        np.random.shuffle(shuffled_contexts) # Shuffle for testing reordering
        
        data = {}
        for i in range(self.sample_num_signatures):
            data[f'Signature_{i+1}'] = np.random.rand(self.sample_num_contexts)
        
        self.sample_profiles_df_shuffled = pd.DataFrame(data, index=shuffled_contexts).T # Signatures as rows, contexts as columns
        self.sample_profiles_df_ordered = self.sample_profiles_df_shuffled[STANDARD_96_CONTEXTS]


        # Sample command line arguments
        self.sample_args_dict = {
            'signature_profiles_path': 'dummy_profiles.csv',
            'output_dir_figures': 'dummy_output_figs/',
            'file_prefix': 'test_sig_'
        }
        self.sample_args_namespace = argparse.Namespace(**self.sample_args_dict)

    def tearDown(self):
        self.patcher_print.stop()

    @patch('argparse.ArgumentParser.parse_args')
    @patch('pandas.read_csv') # To prevent actual file reading in main
    @patch('scripts.visualize_signatures.plot_signature') # To prevent plotting
    @patch('os.makedirs') # To prevent directory creation
    def test_argument_parsing(self, mock_makedirs, mock_plot_sig, mock_read_csv, mock_parse_args):
        """Test argument parsing logic by checking if main uses the parsed args."""
        mock_parse_args.return_value = self.sample_args_namespace
        # Mock read_csv to avoid FileNotFoundError since 'dummy_profiles.csv' doesn't exist
        mock_read_csv.return_value = self.sample_profiles_df_ordered 

        # Call main. We don't need to mock sys.argv because parse_args itself is mocked.
        visualize_signatures.main()

        # Assert that parse_args was called (implicitly, as main runs)
        mock_parse_args.assert_called_once()
        
        # Check if downstream functions are called with values derived from these args
        mock_read_csv.assert_called_with(self.sample_args_dict['signature_profiles_path'], index_col=0)
        mock_makedirs.assert_called_with(self.sample_args_dict['output_dir_figures'], exist_ok=True)
        # Further checks on how args.file_prefix is used are in test_plotting_loop_and_save

    @patch('pandas.read_csv')
    @patch('os.makedirs') # Mock to prevent actual directory creation
    @patch('scripts.visualize_signatures.plot_signature') # Mock to prevent actual plotting
    def test_data_loading_and_reordering(self, mock_plot_signature_func, mock_os_makedirs, mock_pd_read_csv):
        """Test data loading and reordering of contexts."""
        mock_pd_read_csv.return_value = self.sample_profiles_df_shuffled.copy()

        # To test the data loading and reordering part of main(), we need to control
        # the args passed to it, and mock functions called after reordering.
        with patch('argparse.ArgumentParser.parse_args', return_value=self.sample_args_namespace):
            visualize_signatures.main()

        mock_pd_read_csv.assert_called_with(self.sample_args_dict['signature_profiles_path'], index_col=0)
        
        # The script's main function reorders df_profiles internally.
        # The plot_signature mock will receive this reordered data.
        # We check the columns of the 'signature_series' argument passed to the first call
        # of the mocked plot_signature function.
        self.assertTrue(mock_plot_signature_func.called)
        first_call_args = mock_plot_signature_func.call_args_list[0]
        # args[0] is signature_series in plot_signature(signature_series, ...)
        # kwargs['signature_series'] if called with kwargs
        passed_signature_series = first_call_args[1]['signature_series'] # Accessing via kwargs as per definition
        
        self.assertIsInstance(passed_signature_series, pd.Series)
        self.assertEqual(passed_signature_series.index.tolist(), STANDARD_96_CONTEXTS)


    @patch('pandas.read_csv')
    @patch('os.makedirs')
    @patch('scripts.visualize_signatures.plot_signature')
    def test_plotting_loop_and_save(self, mock_plot_signature_func, mock_os_makedirs, mock_pd_read_csv):
        """Test the loop that iterates through signatures and calls the plotting function."""
        mock_pd_read_csv.return_value = self.sample_profiles_df_ordered.copy() # Already ordered

        with patch('argparse.ArgumentParser.parse_args', return_value=self.sample_args_namespace):
            visualize_signatures.main()

        mock_os_makedirs.assert_called_with(self.sample_args_dict['output_dir_figures'], exist_ok=True)
        self.assertEqual(mock_plot_signature_func.call_count, self.sample_num_signatures)

        for i, signature_name_expected in enumerate(self.sample_profiles_df_ordered.index):
            actual_call = mock_plot_signature_func.call_args_list[i]
            
            # Check kwargs of the call, as plot_signature is called with keyword arguments in main
            kwargs = actual_call[1] 

            self.assertIsInstance(kwargs['signature_series'], pd.Series)
            # Check if the passed series matches the expected row from the DataFrame
            pd.testing.assert_series_equal(kwargs['signature_series'], self.sample_profiles_df_ordered.iloc[i], check_names=False)


            expected_plot_title = f"{self.sample_args_dict['file_prefix']}{signature_name_expected}"
            self.assertEqual(kwargs['signature_name'], expected_plot_title)

            safe_sig_name = str(signature_name_expected).replace(" ", "_").replace("/", "_")
            expected_filename = f"{self.sample_args_dict['file_prefix']}{safe_sig_name}.png"
            expected_output_path = os.path.join(self.sample_args_dict['output_dir_figures'], expected_filename)
            self.assertEqual(kwargs['output_path'], expected_output_path)
            
            self.assertEqual(kwargs['colors_map'], MUTATION_TYPE_COLORS)
            self.assertEqual(kwargs['ordered_contexts'], STANDARD_96_CONTEXTS) # Should pass the globally defined one
            self.assertEqual(kwargs['mutation_types_order'], MUTATION_TYPES)


    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.bar')
    @patch('matplotlib.pyplot.title') # Corrected from pyplot.title to module level
    @patch('matplotlib.pyplot.ylabel')
    @patch('matplotlib.pyplot.xlabel') # Added for completeness from script
    @patch('matplotlib.pyplot.xticks')
    @patch('matplotlib.pyplot.text')   # For the group headers
    @patch('matplotlib.pyplot.axvline') # For the group separators
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.figure', return_value=(MagicMock(), MagicMock())) # Mocks fig, ax
    def test_plot_signature_function_calls(self, mock_figure_constructor, mock_close, 
                                            mock_axvline, mock_text, mock_xticks, mock_xlabel,
                                            mock_ylabel, mock_title, mock_bar, mock_savefig):
        """Test detailed calls within the plot_signature function."""
        
        # Get the mocked ax from the figure mock
        mock_fig, mock_ax = mock_figure_constructor.return_value
        # Configure ax.get_ylim() for the text positioning logic
        mock_ax.get_ylim.return_value = (0, 0.1) # Example y-limits

        sample_series_data = pd.Series(np.random.rand(self.sample_num_contexts), index=STANDARD_96_CONTEXTS)
        test_signature_name = "TestSig1"
        dummy_output_path = "dummy_output/TestSig1.png"

        visualize_signatures.plot_signature(
            signature_series=sample_series_data,
            signature_name=test_signature_name,
            output_path=dummy_output_path,
            colors_map=MUTATION_TYPE_COLORS,
            ordered_contexts=STANDARD_96_CONTEXTS,
            mutation_types_order=MUTATION_TYPES
        )

        mock_figure_constructor.assert_called_once_with(figsize=(22, 11))
        mock_bar.assert_called_once()
        # Check the data passed to bar: first arg is x (contexts), second is y (series values)
        self.assertEqual(len(mock_bar.call_args[0][0]), self.sample_num_contexts) # x values (contexts)
        self.assertEqual(len(mock_bar.call_args[0][1]), self.sample_num_contexts) # y values (series data)
        
        mock_title.assert_called_once_with(test_signature_name, fontsize=18, pad=35)
        mock_ylabel.assert_called_once_with("Contribution", fontsize=16)
        mock_xlabel.assert_called_once_with("Trinucleotide Context", fontsize=16, labelpad=10)
        
        # xticks are complex: set_xticks then set_xticklabels
        # Check if set_xticks was called with range(96)
        # mock_ax.set_xticks.assert_called_once_with(range(self.sample_num_contexts)) 
        # This assertion needs mock_ax to be the one used.
        # The call to mock_xticks directly patches `plt.xticks` which isn't how it's used with an Axes object.
        # The script uses ax.set_xticks. So, we need to check the mock_ax.
        mock_ax.set_xticks.assert_called_once_with(range(self.sample_num_contexts))
        mock_ax.set_xticklabels.assert_called_once() # Check it was called
        
        # Check vertical lines for group separators (5 lines for 6 groups)
        self.assertEqual(mock_ax.axvline.call_count, 5)
        
        # Check text headers for mutation types (6 headers)
        self.assertEqual(mock_ax.text.call_count, 6)

        mock_savefig.assert_called_once_with(dummy_output_path, dpi=300)
        mock_close.assert_called_once_with(mock_fig)


if __name__ == '__main__':
    unittest.main()
