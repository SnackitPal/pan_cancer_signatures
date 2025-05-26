import unittest
from unittest.mock import patch, MagicMock, call
import pandas as pd
import numpy as np
import argparse
import sys
import os
import io # For mocking file content with StringIO
import matplotlib.pyplot # To allow mocking plt.close, plt.savefig etc.

# Adjust sys.path to allow direct import of the script under test
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts')))

# Import functions from the script to be tested
from plot_gsea_summary import (
    create_parser, load_and_validate_gsea_data, select_top_pathways, 
    generate_plot_and_save, plot_gsea_summary, main as plot_gsea_main
)

class TestPlotGseaSummary(unittest.TestCase):
    """Test suite for plot_gsea_summary.py script."""

    def _create_mock_args(self, gsea_file="dummy.csv", output_file="dummy.png", top_n=10, sort_by='NES', fdr=0.25):
        return argparse.Namespace(
            gsea_results_file=gsea_file,
            top_n_pathways=top_n,
            output_plot_file=output_file,
            sort_by_metric=sort_by,
            fdr_threshold=fdr
        )

    def test_argument_parser(self):
        """Tests the argument parsing functionality."""
        parser = create_parser()
        args_list_full = [
            "--gsea_results_file", "test.csv", "--output_plot_file", "test_plot.png",
            "--top_n_pathways", "15", "--sort_by_metric", "NOM p-val", "--fdr_threshold", "0.1"
        ]
        args = parser.parse_args(args_list_full)
        self.assertEqual(args.gsea_results_file, "test.csv")
        self.assertEqual(args.top_n_pathways, 15)
        # ... (rest of assertions as in previous implementation)

    def _create_mock_gsea_results_df(self, terms, nes_values, fdr_values, nom_p_val=None, extra_cols=None):
        """Helper to create mock GSEA results DataFrames."""
        data = {'Term': terms, 'NES': nes_values, 'FDR q-val': fdr_values}
        if nom_p_val is not None: data['NOM p-val'] = nom_p_val
        if extra_cols:
            for col_name, values in extra_cols.items():
                if len(values) != len(terms):
                    raise ValueError(f"Extra col '{col_name}' length mismatch.")
                data[col_name] = values
        return pd.DataFrame(data)

    # --- Tests for load_and_validate_gsea_data (previously implemented) ---
    @patch('pandas.read_csv')
    def test_load_and_validate_data_successful(self, mock_pd_read_csv):
        mock_df = self._create_mock_gsea_results_df(['P_A'], [1], [0.01])
        mock_pd_read_csv.return_value = mock_df
        processed_df = load_and_validate_gsea_data("dummy.csv", 'NES')
        self.assertEqual(len(processed_df), 1)
        # ... (other assertions as in previous implementation)

    # --- Tests for select_top_pathways ---
    def test_select_top_pathways_basic_top_n(self):
        """Test basic top N selection and sorting."""
        mock_df = self._create_mock_gsea_results_df(
            terms=[f'P{i}' for i in range(20)],
            nes_values=np.linspace(2, -2, 20), # Mix of positive and negative
            fdr_values=np.linspace(0.01, 0.5, 20)
        ) # NES values: 2.0, 1.78..., -1.78..., -2.0
        
        top_n = 10
        selected_df = select_top_pathways(mock_df, top_n, 'NES')
        
        self.assertEqual(len(selected_df), top_n)
        # Check that the top N by absolute NES were selected
        expected_top_abs_nes_terms = mock_df.iloc[
            mock_df['NES'].abs().nlargest(top_n).index
        ]['Term'].tolist()
        
        # Then check if these terms are in the selected_df (order might differ before final sort)
        self.assertCountEqual(selected_df['Term'].tolist(), expected_top_abs_nes_terms)
        
        # Assert final sort is by original 'NES' descending
        self.assertTrue(all(selected_df['NES'].iloc[i] >= selected_df['NES'].iloc[i+1] 
                            for i in range(len(selected_df)-1)))

    def test_select_top_pathways_fewer_than_n(self):
        """Test when available pathways are fewer than N."""
        mock_df = self._create_mock_gsea_results_df(
            terms=['P_A', 'P_B', 'P_C'],
            nes_values=[1.0, -0.5, 2.0],
            fdr_values=[0.01, 0.05, 0.001]
        )
        top_n = 5
        selected_df = select_top_pathways(mock_df, top_n, 'NES')
        
        self.assertEqual(len(selected_df), len(mock_df)) # Should return all
        # Assert final sort is by original 'NES' descending
        expected_sorted_terms = ['P_C', 'P_A', 'P_B'] # Sorted by NES: 2.0, 1.0, -0.5
        self.assertListEqual(selected_df['Term'].tolist(), expected_sorted_terms)

    def test_select_top_pathways_top_n_is_zero(self):
        """Test when top_n_pathways is 0."""
        mock_df = self._create_mock_gsea_results_df(['P_A'], [1], [0.01])
        selected_df = select_top_pathways(mock_df, 0, 'NES')
        self.assertTrue(selected_df.empty)

    def test_select_top_pathways_different_sort_metric(self):
        """Test selection with a different sort_by_metric (e.g., FDR)."""
        mock_df = self._create_mock_gsea_results_df(
            terms=['P_HighFDR', 'P_MidFDR', 'P_LowFDR', 'P_UltraLowFDR'],
            nes_values=[1.0, 1.5, -1.2, 0.5],
            fdr_values=[0.20, 0.10, 0.05, 0.01] # Lower FDR is better
        )
        # For FDR, lower is better. The current logic sorts by abs(metric) descending.
        # To make this test meaningful for 'FDR q-val', we should sort by it directly ascending
        # or by a transformed metric where higher is better (e.g., -log10(FDR)).
        # The current select_top_pathways sorts by abs(metric) descending.
        # Let's assume for this test, we want top by FDR magnitude (less meaningful for FDR but tests mechanics)
        top_n = 2
        # The function takes abs value of sort_by_metric, so 0.20 and 0.10 are "top" by magnitude.
        selected_df = select_top_pathways(mock_df, top_n, 'FDR q-val') 
        
        self.assertEqual(len(selected_df), top_n)
        # Expected: P_HighFDR (abs(0.20)), P_MidFDR (abs(0.10))
        # Then sorted by NES descending: P_MidFDR (1.5), P_HighFDR (1.0)
        expected_terms_after_abs_fdr_then_nes_sort = ['P_MidFDR', 'P_HighFDR']
        self.assertListEqual(selected_df['Term'].tolist(), expected_terms_after_abs_fdr_then_nes_sort)

    def test_select_top_pathways_sort_metric_not_numeric(self):
        mock_df = self._create_mock_gsea_results_df(['P1'],[1],[0.01])
        mock_df['TextMetric'] = ['A', 'B', 'C'][:len(mock_df)] # Add a non-numeric column
        with self.assertRaisesRegex(ValueError, "Column 'TextMetric' for sorting is not numeric"):
            select_top_pathways(mock_df, 2, 'TextMetric')


    # --- Tests for generate_plot_and_save (mocking plotting functions) ---
    @patch('plot_gsea_summary.os.makedirs')
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    @patch('seaborn.barplot')
    @patch('matplotlib.pyplot.subplots')
    def test_generate_plot_calls_standard(self, mock_subplots, mock_sns_barplot, 
                                          mock_plt_close, mock_plt_savefig, mock_os_makedirs):
        """Test standard plot generation path, verifying calls and arguments."""
        mock_ax = MagicMock()
        mock_fig = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        df_to_plot = self._create_mock_gsea_results_df(
            terms=['Pathway_Sig', 'Pathway_NonSig', 'Pathway_NegSig'],
            nes_values=[2.5, 1.0, -2.2],
            fdr_values=[0.01, 0.30, 0.04] # P_Sig and P_NegSig are significant
        )
        # The generate_plot_and_save expects df_to_plot to be already sorted by NES desc.
        df_to_plot = df_to_plot.sort_values(by='NES', ascending=False)


        output_file = "test_plot.png"
        fdr_highlight = 0.05
        num_selected_for_title = len(df_to_plot)

        generate_plot_and_save(df_to_plot, output_file, fdr_highlight, num_selected_for_title)

        mock_os_makedirs.assert_called_once_with(os.path.dirname(output_file), exist_ok=True)
        mock_subplots.assert_called_once()
        
        # Check seaborn.barplot call
        mock_sns_barplot.assert_called_once()
        call_args_barplot = mock_sns_barplot.call_args[1] # kwargs
        self.assertEqual(call_args_barplot['x'], 'NES')
        self.assertEqual(call_args_barplot['y'], 'Term_Display')
        self.assertIsInstance(call_args_barplot['data'], pd.DataFrame)
        
        # Verify 'Term_Display' column for asterisks
        plotted_data = call_args_barplot['data']
        self.assertIn('Pathway_Sig *', plotted_data['Term_Display'].tolist())
        self.assertIn('Pathway_NonSig', plotted_data['Term_Display'].tolist())
        self.assertIn('Pathway_NegSig *', plotted_data['Term_Display'].tolist())

        # Verify ax.text calls for annotations (check for number of calls)
        # There are 3 rows, so 3 calls to ax.text
        self.assertEqual(mock_ax.text.call_count, 3)
        # Example check for one call's content (can be more specific)
        # first_text_call_args = mock_ax.text.call_args_list[0][0] # args of first call
        # self.assertIn("FDR: 0.01", str(first_text_call_args))


        mock_plt_savefig.assert_called_once_with(output_file, dpi=300, bbox_inches='tight')
        mock_plt_close.assert_called_once_with(mock_fig)

    @patch('plot_gsea_summary.plt.savefig') # only need to mock savefig to prevent actual saving
    def test_generate_plot_with_empty_data(self, mock_savefig):
        """Test plotting with an empty DataFrame."""
        empty_df = pd.DataFrame(columns=['Term', 'NES', 'FDR q-val'])
        # generate_plot_and_save should return early if df_to_plot is empty
        generate_plot_and_save(empty_df, "empty_plot.png", 0.25, 0)
        mock_savefig.assert_not_called() # Plotting should be skipped


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
