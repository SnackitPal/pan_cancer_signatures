import unittest
from unittest import mock
from unittest.mock import MagicMock, patch, mock_open, call
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

import scripts.analyze_patient_exposures as analyze_patient_exposures # Script to be tested

class TestAnalyzePatientExposures(unittest.TestCase):

    def setUp(self):
        """Set up for test methods."""
        self.sample_exposures_data = {
            'Tumor_Sample_Barcode': ['S1', 'S2', 'S3', 'S4'], # S4 will be missing from map
            'Signature_1': [0.1, 0.2, 0.3, 0.4], 
            'Signature_2': [0.9, 0.8, 0.7, 0.6]
        }
        self.sample_exposures_df = pd.DataFrame(self.sample_exposures_data)
        # The script expects index_col=0 for exposures, so we set it for the mock input
        self.sample_exposures_df_indexed = self.sample_exposures_df.set_index('Tumor_Sample_Barcode')

        self.sample_cohort_map_data = {
            'Tumor_Sample_Barcode': ['S1', 'S2', 'S3', 'S5'], # S5 not in exposures
            'Cohort': ['LUAD', 'LUAD', 'SKCM', 'BRCA']
        }
        self.sample_cohort_map_df = pd.DataFrame(self.sample_cohort_map_data)

        self.default_args = argparse.Namespace(
            exposures_file='dummy_exposures.csv',
            sample_cohort_map_file='dummy_cohort_map.csv',
            output_dir_figures='dummy_output_figs/'
        )

        # Mock sys.stdout to capture print statements
        self.patcher_stdout = patch('sys.stdout', new_callable=io.StringIO)
        
        # Suppress print statements from the script itself (if any are not captured by stdout mock)
        self.patcher_bprint = patch('builtins.print') 
        self.mock_bprint = self.patcher_bprint.start()

    def tearDown(self):
        self.patcher_bprint.stop()
        # Ensure stdout patcher is stopped if started
        if hasattr(self, 'mock_stdout') and self.mock_stdout.is_started: # Check if mock_stdout was started
             self.mock_stdout.stop()


    @patch('argparse.ArgumentParser.parse_args')
    @patch('pandas.read_csv', side_effect=FileNotFoundError) # Mock to prevent file ops
    @patch('sys.exit') 
    def test_argument_parsing(self, mock_sys_exit, mock_read_csv, mock_parse_args):
        """Test argument parsing logic by checking if main uses the parsed args."""
        test_args_namespace = argparse.Namespace(
            exposures_file='path_exp.csv',
            sample_cohort_map_file='path_map.csv',
            output_dir_figures='out_figs/'
        )
        mock_parse_args.return_value = test_args_namespace

        with patch('sys.argv', ['analyze_patient_exposures.py', '--exposures_file', 'path_exp.csv', 
                                '--sample_cohort_map_file', 'path_map.csv', 
                                '--output_dir_figures', 'out_figs/']):
            analyze_patient_exposures.main()

        mock_parse_args.assert_called_once()
        # Check pd.read_csv calls
        mock_read_csv.assert_any_call('path_exp.csv', index_col=0)
        # The side_effect=FileNotFoundError will prevent the second call, which is fine for this test's scope.


    @patch('pandas.read_csv')
    @patch('os.makedirs') # Mock to prevent actual directory creation
    @patch('scripts.analyze_patient_exposures.sns.boxplot') # Mock actual plotting
    @patch('scripts.analyze_patient_exposures.plt.savefig') 
    @patch('scripts.analyze_patient_exposures.plt.close')
    def test_data_loading_and_merging(self, mock_plt_close, mock_plt_savefig, mock_sns_boxplot, 
                                     mock_os_makedirs, mock_pd_read_csv):
        """Test data loading, preparation, and merging logic."""
        self.mock_stdout = self.patcher_stdout.start()

        mock_pd_read_csv.side_effect = [
            self.sample_exposures_df_indexed.copy(), 
            self.sample_cohort_map_df.copy()
        ]

        with patch('argparse.ArgumentParser.parse_args', return_value=self.default_args):
            analyze_patient_exposures.main()

        self.assertEqual(mock_pd_read_csv.call_count, 2)
        mock_pd_read_csv.assert_any_call(self.default_args.exposures_file, index_col=0)
        mock_pd_read_csv.assert_any_call(self.default_args.sample_cohort_map_file)
        
        output = self.mock_stdout.getvalue()
        self.assertIn("Warning: 1 samples from exposures file were not found in the cohort map.", output)
        self.assertIn("- S4", output) # S4 is the sample missing from cohort map
        self.assertIn("Dropped 1 samples with missing cohort information.", output)

        # Verify the merged DataFrame used for plotting (via boxplot call)
        # Expected merged data: S1, S2, S3 with their cohorts. S4 dropped. S5 ignored.
        self.assertTrue(mock_sns_boxplot.called)
        # The 'data' argument to boxplot is the merged_df
        passed_to_boxplot_df = mock_sns_boxplot.call_args_list[0][1]['data'] # Accessing via kwargs
        
        self.assertEqual(len(passed_to_boxplot_df), 3) # S1, S2, S3 remain
        self.assertListEqual(sorted(passed_to_boxplot_df['Tumor_Sample_Barcode'].tolist()), ['S1', 'S2', 'S3'])
        self.assertNotIn('S4', passed_to_boxplot_df['Tumor_Sample_Barcode'].tolist())
        self.assertNotIn('S5', passed_to_boxplot_df['Tumor_Sample_Barcode'].tolist())
        self.assertListEqual(sorted(passed_to_boxplot_df['Cohort'].unique().tolist()), ['LUAD', 'SKCM'])
        
        self.patcher_stdout.stop()


    @patch('pandas.read_csv')
    @patch('seaborn.boxplot')
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.title')
    @patch('os.makedirs')
    def test_boxplots_generation(self, mock_os_makedirs, mock_plt_title, mock_plt_close, 
                                mock_plt_savefig, mock_sns_boxplot, mock_pd_read_csv):
        """Test boxplot generation loop and saving."""
        # Prepare a merged DataFrame that would be the result of loading and merging
        merged_df_for_test = pd.merge(
            self.sample_exposures_df, 
            self.sample_cohort_map_df, 
            on='Tumor_Sample_Barcode', 
            how='left'
        ).dropna(subset=['Cohort'])
        
        mock_pd_read_csv.side_effect = [
            self.sample_exposures_df_indexed.copy(), 
            self.sample_cohort_map_df.copy()
        ]

        with patch('argparse.ArgumentParser.parse_args', return_value=self.default_args):
            analyze_patient_exposures.main()
            
        mock_os_makedirs.assert_called_with(self.default_args.output_dir_figures, exist_ok=True)

        num_signatures = 2 # Signature_1, Signature_2
        self.assertEqual(mock_sns_boxplot.call_count, num_signatures)
        self.assertEqual(mock_plt_savefig.call_count, num_signatures)
        self.assertEqual(mock_plt_close.call_count, num_signatures)

        for i, sig_name in enumerate(['Signature_1', 'Signature_2']):
            # Check boxplot call
            boxplot_call_args = mock_sns_boxplot.call_args_list[i][1] # kwargs
            self.assertEqual(boxplot_call_args['x'], 'Cohort')
            self.assertEqual(boxplot_call_args['y'], sig_name)
            pd.testing.assert_frame_equal(boxplot_call_args['data'], merged_df_for_test, check_dtype=False)

            # Check savefig call
            savefig_call_args = mock_plt_savefig.call_args_list[i][0] # args
            expected_filename = os.path.join(self.default_args.output_dir_figures, f"exposure_boxplot_{sig_name}.png")
            self.assertEqual(savefig_call_args[0], expected_filename)

    @patch('pandas.read_csv')
    @patch('scripts.analyze_patient_exposures.plt.figure') # Mock away all plotting
    @patch('scripts.analyze_patient_exposures.sns.boxplot')
    @patch('scripts.analyze_patient_exposures.plt.savefig')
    @patch('scripts.analyze_patient_exposures.plt.close')
    @patch('scripts.analyze_patient_exposures.pd.DataFrame.plot') # Mock the df.plot for stacked bar
    def test_summary_statistics_calculation(self, mock_df_plot, mock_plt_close, mock_plt_savefig, 
                                          mock_sns_boxplot, mock_plt_figure, mock_pd_read_csv):
        """Test calculation and printing of summary statistics."""
        self.mock_stdout = self.patcher_stdout.start()
        
        mock_pd_read_csv.side_effect = [
            self.sample_exposures_df_indexed.copy(), 
            self.sample_cohort_map_df.copy()
        ]

        with patch('argparse.ArgumentParser.parse_args', return_value=self.default_args):
            analyze_patient_exposures.main()

        output = self.mock_stdout.getvalue()
        self.assertIn("Summary Statistics of Signature Exposures per Cohort:", output)
        
        # Expected means for Signature_1:
        # LUAD: (0.1 + 0.2) / 2 = 0.15
        # SKCM: 0.3 / 1 = 0.3
        self.assertIn("Signature_1", output) # Check if column group is present
        self.assertIn("mean", output)       # Check if 'mean' row is present
        self.assertIn("median", output)     # Check if 'median' row is present
        
        # A more robust check would parse the table from stdout or have the function return the stats df.
        # For this test, checking for key parts of the expected output.
        # Example: searching for LUAD stats for Signature_1
        # This is highly dependent on pandas' df.to_string() format.
        self.assertRegex(output, r"LUAD\s+0\.15") # Mean for Sig1 in LUAD
        self.assertRegex(output, r"SKCM\s+0\.30") # Mean for Sig1 in SKCM

        self.patcher_stdout.stop()


    @patch('pandas.read_csv')
    @patch('matplotlib.pyplot.figure') # Mock to control figure creation
    @patch('pandas.DataFrame.plot') # Mock the plot method on DataFrame
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.title')
    @patch('matplotlib.pyplot.xlabel')
    @patch('matplotlib.pyplot.ylabel')
    @patch('matplotlib.pyplot.xticks')
    @patch('matplotlib.pyplot.legend')
    @patch('os.makedirs')
    def test_stacked_bar_plot_generation(self, mock_os_makedirs, mock_plt_legend, mock_plt_xticks, 
                                         mock_plt_ylabel, mock_plt_xlabel, mock_plt_title, 
                                         mock_plt_close, mock_plt_savefig, mock_df_plot, 
                                         mock_plt_figure, mock_pd_read_csv):
        """Test generation of the stacked bar plot."""
        mock_pd_read_csv.side_effect = [
            self.sample_exposures_df_indexed.copy(), 
            self.sample_cohort_map_df.copy()
        ]
        # Mock the plot method to return a mock Axes object, as the script uses `ax = ...plot()`
        mock_ax = MagicMock()
        mock_df_plot.return_value = mock_ax


        with patch('argparse.ArgumentParser.parse_args', return_value=self.default_args):
            analyze_patient_exposures.main()

        # Check that mean_exposures_by_cohort.plot() was called
        mock_df_plot.assert_called_once()
        call_kwargs = mock_df_plot.call_args[1]
        self.assertEqual(call_kwargs['kind'], 'bar')
        self.assertTrue(call_kwargs['stacked'])
        self.assertIn('figsize', call_kwargs) # Check if figsize is passed
        
        mock_plt_title.assert_called_with("Average Signature Contributions per Cancer Cohort", fontsize=16)
        mock_plt_xlabel.assert_called_with("Cancer Cohort", fontsize=12)
        mock_plt_ylabel.assert_called_with("Mean Signature Exposure", fontsize=12)
        mock_plt_xticks.assert_called_with(rotation=45, ha='right')
        mock_plt_legend.assert_called() # Check it's called, specific args can be complex

        expected_filename = os.path.join(self.default_args.output_dir_figures, "avg_signature_contributions_by_cohort.png")
        mock_plt_savefig.assert_called_once_with(expected_filename, dpi=300, bbox_inches='tight')
        mock_plt_close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
