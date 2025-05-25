import unittest
from unittest import mock # Use this for patch directly
from unittest.mock import MagicMock, patch, mock_open # mock_open is not listed but can be useful
import pandas as pd
import numpy as np
import argparse
import sys
import os
import joblib # For checking joblib.dump calls

# Add scripts directory to sys.path to allow direct import
scripts_dir = os.path.join(os.path.dirname(__file__), '..', 'scripts')
if scripts_dir not in sys.path:
    sys.path.insert(0, os.path.abspath(scripts_dir))

import train_lda_model # Import the script to be tested

class TestTrainLdaModel(unittest.TestCase):

    def setUp(self):
        # Suppress print statements from the script during tests for cleaner output
        self.patcher_print = patch('builtins.print')
        self.mock_print = self.patcher_print.start()

    def tearDown(self):
        self.patcher_print.stop()

    def test_parse_args(self):
        """Test argument parsing logic."""
        # The script's main function handles argparse. We'll mock sys.argv.
        test_argv = [
            'scripts/train_lda_model.py',
            '--matrix_path', 'test_matrix.csv',
            '--k', '5',
            '--output_dir_models', 'test_output_models/',
            '--random_seed', '123'
        ]
        
        # To test argument parsing isolated from the rest of main(),
        # we can re-create the parser or directly call main and check its args usage.
        # For this test, we'll check if main receives these args correctly by mocking a downstream function
        # that would use these args, or by having main return args (if refactored).
        # Here, we'll patch functions that use these arguments.
        
        # Let's assume the script's main function directly uses the parsed args.
        # We will patch 'pandas.read_csv' as an indicator that main proceeded with parsed args.
        with patch('sys.argv', test_argv), \
             patch('pandas.read_csv', side_effect=FileNotFoundError) as mock_read_csv, \
             patch('sys.exit') as mock_exit: # Mock exit to prevent test termination

            train_lda_model.main() # Call main which will parse args

            # Check that read_csv was called with the path from args
            # This indirectly verifies that --matrix_path was parsed.
            mock_read_csv.assert_called_with('test_matrix.csv')
            
            # To directly test other args like k and random_seed, we'd need to mock
            # LatentDirichletAllocation or other functions that receive them.
            # This is covered in test_train_lda_and_save_outputs.
            # For a pure argparse test, one might define a parse_arguments function in the script.

    @patch('pandas.read_csv')
    @patch('sys.exit') # Mock sys.exit to check for early exits on validation errors
    def test_load_and_prepare_data(self, mock_sys_exit, mock_pd_read_csv):
        """Test data loading, preparation, and validation."""
        
        # 1. Successful data loading
        sample_ids = [f'Sample{i+1}' for i in range(3)]
        mock_data = {'SampleID': sample_ids}
        # Create 96 feature columns
        for i in range(96):
            mock_data[f'Context{i+1}'] = np.random.randint(0, 100, size=3)
        
        mock_df = pd.DataFrame(mock_data)
        mock_pd_read_csv.return_value = mock_df.copy() # Return a copy

        # Dummy args for this part of main
        args = argparse.Namespace(
            matrix_path='dummy_path.csv',
            k=3, output_dir_models='out', random_seed=42 # Other args not relevant here
        )

        # We need to run the data loading part of main.
        # This is hard if not refactored. We'll simulate it or call main and mock downstream.
        with patch('argparse.ArgumentParser.parse_args', return_value=args), \
             patch('scripts.train_lda_model.LatentDirichletAllocation') as MockLDA: # Mock LDA to stop execution there
            
            train_lda_model.main()

        mock_pd_read_csv.assert_called_with('dummy_path.csv')
        
        # The script reassigns mutation_df to mutation_matrix after setting index.
        # We need to check the state of 'mutation_matrix' variable within main or how it's passed.
        # This test design is limited by not having refactored main().
        # Assuming the print statements reflect the shape:
        self.mock_print.assert_any_call("Set 'SampleID' as index.")
        self.mock_print.assert_any_call(f"Loaded mutation matrix with shape: {(3, 96)}")


        # 2. Test negative value check
        mock_df_negative = mock_df.copy()
        mock_df_negative.iloc[0, 1] = -5 # Introduce a negative value (index 1 because index 0 is SampleID)
        mock_pd_read_csv.return_value = mock_df_negative
        
        # Reset mock_sys_exit for this specific test case
        mock_sys_exit.reset_mock()

        with patch('argparse.ArgumentParser.parse_args', return_value=args), \
             patch('scripts.train_lda_model.LatentDirichletAllocation'): # Mock LDA again
            train_lda_model.main()
        
        self.mock_print.assert_any_call("Error: Mutation matrix contains negative values. LDA expects non-negative counts.")
        # mock_sys_exit.assert_called_once() # The script uses 'return', not 'sys.exit' for this.
        # So, we check if the error message was printed. The function should have returned.

    @patch('pandas.read_csv')
    @patch('scripts.train_lda_model.LatentDirichletAllocation')
    @patch('os.makedirs')
    @patch('joblib.dump')
    @patch('pandas.DataFrame.to_csv') # Mock the to_csv method of DataFrame instances
    def test_train_lda_and_save_outputs(self, mock_df_to_csv, mock_joblib_dump, mock_os_makedirs, 
                                        MockLDAClass, mock_pd_read_csv):
        """Test LDA training, model fitting, and output saving."""
        
        NUM_SAMPLES = 5
        NUM_FEATURES = 96 # Standard 96 contexts
        NUM_SIGNATURES_K = 3
        RANDOM_SEED = 42
        OUTPUT_DIR = 'test_output_dir'
        MATRIX_PATH = 'dummy_matrix.csv'

        # Prepare mock DataFrame for read_csv
        sample_ids = [f'Sample{i+1}' for i in range(NUM_SAMPLES)]
        mock_contexts = [f'Context{i+1}' for i in range(NUM_FEATURES)]
        data_for_df = {'SampleID': sample_ids}
        for i in range(NUM_FEATURES):
            data_for_df[mock_contexts[i]] = np.random.randint(0, 100, size=NUM_SAMPLES)
        mock_input_df = pd.DataFrame(data_for_df)
        mock_pd_read_csv.return_value = mock_input_df.copy()

        # Configure MockLDAClass and its instance
        mock_lda_instance = MockLDAClass.return_value
        mock_lda_instance.components_ = np.random.rand(NUM_SIGNATURES_K, NUM_FEATURES)
        mock_lda_instance.transform.return_value = np.random.rand(NUM_SAMPLES, NUM_SIGNATURES_K)

        # Prepare args for main()
        # Note: if main() is not modified to accept argv, we patch sys.argv
        dummy_sys_argv = [
            'scripts/train_lda_model.py',
            '--matrix_path', MATRIX_PATH,
            '--k', str(NUM_SIGNATURES_K),
            '--output_dir_models', OUTPUT_DIR,
            '--random_seed', str(RANDOM_SEED)
        ]

        with patch.object(sys, 'argv', dummy_sys_argv):
            train_lda_model.main()

        # Assertions
        MockLDAClass.assert_called_once_with(
            n_components=NUM_SIGNATURES_K,
            random_state=RANDOM_SEED,
            learning_method='batch',
            max_iter=10 
        )
        mock_lda_instance.fit.assert_called_once()
        # Check that the argument to fit was a DataFrame or numpy array of the correct shape
        self.assertEqual(mock_lda_instance.fit.call_args[0][0].shape, (NUM_SAMPLES, NUM_FEATURES))
        
        mock_lda_instance.transform.assert_called_once()
        self.assertEqual(mock_lda_instance.transform.call_args[0][0].shape, (NUM_SAMPLES, NUM_FEATURES))

        mock_os_makedirs.assert_called_with(OUTPUT_DIR, exist_ok=True)

        expected_model_filename = os.path.join(OUTPUT_DIR, f"lda_model_k{NUM_SIGNATURES_K}_seed{RANDOM_SEED}.joblib")
        mock_joblib_dump.assert_called_once_with(mock_lda_instance, expected_model_filename)

        self.assertEqual(mock_df_to_csv.call_count, 2)

        # Inspect calls to DataFrame.to_csv()
        # Call 0: Signature Profiles
        profiles_call_args = mock_df_to_csv.call_args_list[0]
        expected_profiles_filename = os.path.join(OUTPUT_DIR, f"signature_profiles_k{NUM_SIGNATURES_K}_seed{RANDOM_SEED}.csv")
        self.assertEqual(profiles_call_args[0][0], expected_profiles_filename)
        
        # The DataFrame instance itself is `profiles_call_args.instance` if needed,
        # but we check based on how df_profiles is created in the script.
        # The script calls `df_profiles.to_csv(...)`. The first arg to to_csv is the path.
        # To check the DataFrame content, we'd need to capture the instance `self` for that call.
        # This is tricky. Instead, we trust the script creates it correctly and focus on filename and that it was called.
        # However, we can access the `self` of the mocked method if we assign the mock differently.
        # For now, let's assume the structure based on script logic:
        # df_profiles = pd.DataFrame(profiles_normalized, index=[Signature_...], columns=mutation_matrix.columns)
        # So, the columns should be the context names from the input, and index the signature names.
        
        # Call 1: Patient Exposures
        exposures_call_args = mock_df_to_csv.call_args_list[1]
        expected_exposures_filename = os.path.join(OUTPUT_DIR, f"patient_exposures_k{NUM_SIGNATURES_K}_seed{RANDOM_SEED}.csv")
        self.assertEqual(exposures_call_args[0][0], expected_exposures_filename)
        # df_exposures = pd.DataFrame(exposures, index=mutation_matrix.index, columns=[Signature_...])
        # Index should be sample IDs from input, columns are signature names.

if __name__ == '__main__':
    unittest.main()
