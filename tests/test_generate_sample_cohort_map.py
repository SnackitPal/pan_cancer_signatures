import unittest
from unittest import mock
from unittest.mock import MagicMock, patch, mock_open, call
import pandas as pd
import argparse
import sys
import os
import io # For StringIO

# Add scripts directory to sys.path to allow direct import
scripts_dir = os.path.join(os.path.dirname(__file__), '..', 'scripts')
if scripts_dir not in sys.path:
    sys.path.insert(0, os.path.abspath(scripts_dir))

import scripts.generate_sample_cohort_map as generate_sample_cohort_map

class TestGenerateSampleCohortMap(unittest.TestCase):

    def setUp(self):
        """Set up for test methods."""
        self.maf_df1_cohort1 = pd.DataFrame({'Tumor_Sample_Barcode': ['Sample1', 'Sample2', 'Sample1']})
        self.maf_df2_cohort1 = pd.DataFrame({'Sample_ID': ['Sample2', 'Sample3']}) # Test alternative column name
        self.maf_df1_cohort2 = pd.DataFrame({'Tumor_Sample_Barcode': ['Sample4', 'Sample1']}) # Sample1 appears in Cohort2

        # Mock sys.stdout to capture print statements
        self.patcher_stdout = patch('sys.stdout', new_callable=io.StringIO)
        # Start it in each test method where needed, or here and manage reset if preferred.
        # For simplicity, starting it in each test method.

        # Suppress print statements from the script itself (if any are not captured by stdout mock)
        self.patcher_bprint = patch('builtins.print')
        self.mock_bprint = self.patcher_bprint.start()

        self.default_args = argparse.Namespace(
            maf_input_dir='dummy_maf_dir/',
            output_map_file='dummy_output_map.csv'
        )

    def tearDown(self):
        self.patcher_bprint.stop()
        # Ensure stdout patcher is stopped if started
        if hasattr(self, 'mock_stdout') and self.mock_stdout.is_started:
             self.mock_stdout.stop()


    @patch('argparse.ArgumentParser.parse_args')
    @patch('os.path.isdir') # Mock to control flow after parsing
    @patch('os.path.exists', return_value=True) # Assume path exists for parsing test
    @patch('os.listdir', return_value=[]) # Assume empty dir for parsing test
    def test_argument_parsing(self, mock_listdir, mock_path_exists, mock_os_isdir, mock_parse_args):
        """Test argument parsing logic by checking if main uses the parsed args."""
        test_args_namespace = argparse.Namespace(
            maf_input_dir='specific_maf_dir/',
            output_map_file='specific_output.csv'
        )
        mock_parse_args.return_value = test_args_namespace
        mock_os_isdir.return_value = True # For the base MAF dir

        # Call main. sys.argv mocking not needed as parse_args is mocked.
        generate_sample_cohort_map.main()

        mock_parse_args.assert_called_once()
        # Check if os.listdir (first significant fs call after parsing and validation)
        # is called with the path from the mocked args.
        mock_listdir.assert_called_with('specific_maf_dir/')


    @patch('os.path.isdir')
    @patch('glob.glob')
    @patch('pandas.read_csv')
    @patch('os.listdir')
    @patch('pandas.DataFrame.to_csv')
    @patch('os.path.exists', return_value=True) # Ensure base MAF dir exists
    def test_directory_iteration_and_maf_processing(self, mock_path_exists, mock_to_csv, 
                                                    mock_os_listdir, mock_read_csv, 
                                                    mock_glob_glob, mock_os_isdir):
        """Test directory iteration, MAF processing, and final CSV output."""
        self.mock_stdout = self.patcher_stdout.start() # Start stdout capture

        # --- Setup Mocks ---
        mock_os_listdir.return_value = ['TCGA-LUAD', 'TCGA-SKCM', 'not_a_dir.txt']

        def isdir_side_effect(path):
            if path.endswith('TCGA-LUAD') or path.endswith('TCGA-SKCM'):
                return True
            if path == self.default_args.maf_input_dir: # Base MAF dir
                return True
            return False
        mock_os_isdir.side_effect = isdir_side_effect

        def glob_side_effect(pattern):
            if 'TCGA-LUAD' in pattern:
                if pattern.endswith("*.maf.gz"): # Assuming this is the first pattern checked
                    return ['dummy_maf_dir/TCGA-LUAD/luad1.maf.gz', 'dummy_maf_dir/TCGA-LUAD/luad2.maf.gz']
            elif 'TCGA-SKCM' in pattern:
                if pattern.endswith("*.maf.gz"):
                    return ['dummy_maf_dir/TCGA-SKCM/skcm1.maf.gz']
            return [] # Default empty for other patterns
        mock_glob_glob.side_effect = glob_side_effect
        
        def read_csv_side_effect(*args, **kwargs):
            filepath = args[0]
            if kwargs.get('nrows') == 0: # Header read
                if 'luad1.maf.gz' in filepath: return pd.DataFrame(columns=['Tumor_Sample_Barcode'])
                if 'luad2.maf.gz' in filepath: return pd.DataFrame(columns=['Sample_ID'])
                if 'skcm1.maf.gz' in filepath: return pd.DataFrame(columns=['Tumor_Sample_Barcode'])
            else: # Data read
                if 'luad1.maf.gz' in filepath: return self.maf_df1_cohort1
                if 'luad2.maf.gz' in filepath: return self.maf_df2_cohort1
                if 'skcm1.maf.gz' in filepath: return self.maf_df1_cohort2
            return pd.DataFrame() # Should not happen with proper mocking
        mock_read_csv.side_effect = read_csv_side_effect

        # --- Call main ---
        with patch('argparse.ArgumentParser.parse_args', return_value=self.default_args):
            generate_sample_cohort_map.main()

        # --- Assertions ---
        mock_os_listdir.assert_called_once_with(self.default_args.maf_input_dir)
        
        # Check glob calls for each valid cohort directory and for multiple patterns
        # Example: 'dummy_maf_dir/TCGA-LUAD/*.maf.gz'
        # The script tries 4 patterns per cohort.
        self.assertEqual(mock_glob_glob.call_count, 2 * 4) # 2 cohorts, 4 patterns each
        mock_glob_glob.assert_any_call(os.path.join(self.default_args.maf_input_dir, 'TCGA-LUAD', '*.maf.gz'))
        mock_glob_glob.assert_any_call(os.path.join(self.default_args.maf_input_dir, 'TCGA-SKCM', '*.maf.gz'))

        # Check read_csv calls (header + data for each file)
        self.assertEqual(mock_read_csv.call_count, 3 * 2) # 3 files, 2 reads each
        
        # Check the DataFrame passed to to_csv
        mock_to_csv.assert_called_once_with(self.default_args.output_map_file, index=False)
        df_output = mock_to_csv.call_args[0][0]
        
        self.assertIsInstance(df_output, pd.DataFrame)
        self.assertListEqual(df_output.columns.tolist(), ['Tumor_Sample_Barcode', 'Cohort'])
        
        # Expected rows after drop_duplicates(keep='first')
        # Sample1 from luad1 (TCGA-LUAD)
        # Sample2 from luad1 (TCGA-LUAD)
        # Sample3 from luad2 (TCGA-LUAD)
        # Sample4 from skcm1 (TCGA-SKCM)
        # Sample1 from skcm1 (TCGA-SKCM) is dropped because Sample1 already seen in TCGA-LUAD.
        expected_data = [
            {'Tumor_Sample_Barcode': 'Sample1', 'Cohort': 'TCGA-LUAD'},
            {'Tumor_Sample_Barcode': 'Sample2', 'Cohort': 'TCGA-LUAD'},
            {'Tumor_Sample_Barcode': 'Sample3', 'Cohort': 'TCGA-LUAD'},
            {'Tumor_Sample_Barcode': 'Sample4', 'Cohort': 'TCGA-SKCM'},
        ]
        expected_df = pd.DataFrame(expected_data).sort_values(by=['Tumor_Sample_Barcode', 'Cohort']).reset_index(drop=True)
        df_output_sorted = df_output.sort_values(by=['Tumor_Sample_Barcode', 'Cohort']).reset_index(drop=True)
        pd.testing.assert_frame_equal(df_output_sorted, expected_df)

        # Check for duplicate warning in stdout
        output = self.mock_stdout.getvalue()
        self.assertIn("Warning: Found and removed 1 duplicate Tumor_Sample_Barcode entries.", output)
        
        self.patcher_stdout.stop()


    @patch('os.path.isdir')
    @patch('glob.glob')
    @patch('pandas.read_csv')
    @patch('os.listdir')
    @patch('pandas.DataFrame.to_csv') # Mock to prevent actual saving
    @patch('os.path.exists', return_value=True)
    def test_robust_barcode_column_finding(self, mock_path_exists, mock_to_csv, mock_os_listdir, 
                                           mock_read_csv, mock_glob_glob, mock_os_isdir):
        self.mock_stdout = self.patcher_stdout.start()
        
        mock_os_listdir.return_value = ['COHORT_TEST']
        mock_os_isdir.return_value = True # All entries are dirs for simplicity here

        # Test case 1: 'Tumor_Sample_Barcode'
        mock_glob_glob.return_value = ['COHORT_TEST/file1.maf.gz']
        df_with_tsb = pd.DataFrame({'Tumor_Sample_Barcode': ['TSB1', 'TSB2']})
        mock_read_csv.side_effect = [pd.DataFrame(columns=['Tumor_Sample_Barcode']), df_with_tsb] # Header, then data
        
        with patch('argparse.ArgumentParser.parse_args', return_value=self.default_args):
            generate_sample_cohort_map.main()
        
        df_output = mock_to_csv.call_args[0][0]
        self.assertIn('TSB1', df_output['Tumor_Sample_Barcode'].values)
        self.assertIn('TSB2', df_output['Tumor_Sample_Barcode'].values)

        # Test case 2: 'Sample_ID'
        mock_to_csv.reset_mock()
        df_with_sid = pd.DataFrame({'Sample_ID': ['SID1', 'SID2']})
        mock_read_csv.side_effect = [pd.DataFrame(columns=['Sample_ID']), df_with_sid]
        with patch('argparse.ArgumentParser.parse_args', return_value=self.default_args):
            generate_sample_cohort_map.main()
        df_output = mock_to_csv.call_args[0][0]
        self.assertIn('SID1', df_output['Tumor_Sample_Barcode'].values)

        # Test case 3: 'sample_id'
        mock_to_csv.reset_mock()
        df_with_sid_lower = pd.DataFrame({'sample_id': ['sid_lower1', 'sid_lower2']})
        mock_read_csv.side_effect = [pd.DataFrame(columns=['sample_id']), df_with_sid_lower]
        with patch('argparse.ArgumentParser.parse_args', return_value=self.default_args):
            generate_sample_cohort_map.main()
        df_output = mock_to_csv.call_args[0][0]
        self.assertIn('sid_lower1', df_output['Tumor_Sample_Barcode'].values)

        # Test case 4: No known barcode column
        mock_to_csv.reset_mock()
        mock_read_csv.reset_mock() # Important to reset side_effect list
        self.mock_stdout.truncate(0); self.mock_stdout.seek(0)
        df_no_known_col = pd.DataFrame({'UnknownColumn': ['Data1', 'Data2']})
        mock_read_csv.side_effect = [pd.DataFrame(columns=['UnknownColumn']), df_no_known_col]
        
        with patch('argparse.ArgumentParser.parse_args', return_value=self.default_args):
             # Patch to_csv to allow inspection of what happens when no barcodes are found
            with patch('pandas.DataFrame.to_csv') as final_mock_to_csv:
                generate_sample_cohort_map.main()
                # Assert that to_csv was NOT called because no valid barcodes were processed
                final_mock_to_csv.assert_not_called() 

        output = self.mock_stdout.getvalue()
        self.assertIn("Warning: Could not find a suitable barcode column", output)
        self.assertIn("No Tumor_Sample_Barcodes found in any MAF files.", output)
        
        self.patcher_stdout.stop()


    @patch('os.listdir')
    @patch('pandas.DataFrame.to_csv')
    @patch('os.path.isdir', return_value=True) # Mock base MAF dir as valid
    @patch('os.path.exists', return_value=True)
    def test_empty_maf_dir(self, mock_path_exists, mock_os_isdir_base, mock_to_csv, mock_os_listdir):
        """Test behavior when MAF input directory is empty or contains no cohort subdirectories."""
        self.mock_stdout = self.patcher_stdout.start()
        mock_os_listdir.return_value = [] # Empty directory

        with patch('argparse.ArgumentParser.parse_args', return_value=self.default_args):
            generate_sample_cohort_map.main()

        mock_to_csv.assert_not_called()
        output = self.mock_stdout.getvalue()
        self.assertIn("No Tumor_Sample_Barcodes found in any MAF files.", output)
        
        self.patcher_stdout.stop()


    @patch('os.path.isdir', return_value=True) # For base MAF dir and output subdirs
    @patch('os.path.exists', return_value=True) # For base MAF dir
    @patch('glob.glob')
    @patch('pandas.read_csv')
    @patch('os.listdir')
    @patch('pandas.DataFrame.to_csv')
    @patch('os.makedirs')
    def test_output_file_creation(self, mock_os_makedirs, mock_to_csv, mock_os_listdir,
                                  mock_read_csv, mock_glob_glob, mock_path_exists_isdir):
        """Test correct creation of output directory and file saving."""
        mock_os_listdir.return_value = ['COHORT_A'] # One cohort
        mock_glob_glob.return_value = ['COHORT_A/fileA.maf.gz'] # One MAF file
        
        # Simulate header and data read for the one MAF file
        header_df = pd.DataFrame(columns=['Tumor_Sample_Barcode'])
        data_df = pd.DataFrame({'Tumor_Sample_Barcode': ['SampleX']})
        mock_read_csv.side_effect = [header_df, data_df]

        nested_output_path = os.path.join('output_results', 'subdir', 'map_file.csv')
        args_nested_output = argparse.Namespace(
            maf_input_dir='dummy_maf_dir/',
            output_map_file=nested_output_path
        )

        with patch('argparse.ArgumentParser.parse_args', return_value=args_nested_output):
            generate_sample_cohort_map.main()

        expected_output_dir = os.path.dirname(nested_output_path)
        mock_os_makedirs.assert_called_once_with(expected_output_dir, exist_ok=True)
        mock_to_csv.assert_called_once_with(nested_output_path, index=False)


if __name__ == '__main__':
    unittest.main()
