import unittest
from unittest import mock
import argparse
import os
import json
import sys

# Add scripts directory to sys.path to allow direct import of download_tcga_mafs
scripts_dir = os.path.join(os.path.dirname(__file__), '..', 'scripts')
sys.path.insert(0, os.path.abspath(scripts_dir))

import download_tcga_mafs

class TestDownloadTcgaMafs(unittest.TestCase):

    def setUp(self):
        """Set up for test methods."""
        # Suppress print statements from the script
        self.patcher = mock.patch('builtins.print')
        self.mock_print = self.patcher.start()

    def tearDown(self):
        """Tear down after test methods."""
        self.patcher.stop()

    def test_argument_parsing(self):
        """Test parsing of command-line arguments."""
        # We need to mock sys.argv for argparse to work as expected in a test
        with mock.patch('sys.argv', ['download_tcga_mafs.py', '--project_ids', 'TCGA-LUAD,TCGA-SKCM', '--output_dir', './test_data/']):
            parser = download_tcga_mafs.main.__globals__['parser'] # Access parser from main
            args = parser.parse_args()
            self.assertEqual(args.project_ids, 'TCGA-LUAD,TCGA-SKCM')
            self.assertEqual(args.output_dir, './test_data/')
            
            # Test the processing of project_ids string into a list
            project_ids_list = [pid.strip() for pid in args.project_ids.split(',')]
            self.assertEqual(project_ids_list, ['TCGA-LUAD', 'TCGA-SKCM'])


    @mock.patch('scripts.download_tcga_mafs.requests.get')
    def test_gdc_api_query_construction(self, mock_requests_get):
        """Test construction of GDC API query parameters."""
        # Mock a minimal successful response for the files endpoint
        mock_response = mock.Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": {
                "hits": [], # No files needed for this test, just checking params
                "pagination": {"total": 0, "count": 0, "page": 1, "pages": 1, "from": 0, "size": 100}
            }
        }
        mock_requests_get.return_value = mock_response

        # Call the main function with dummy arguments to trigger API call
        test_args = argparse.Namespace(project_ids='TCGA-TEST', output_dir='./fake_dir')
        
        # To test the params, we need to run part of the main logic.
        # We can't directly call main() as it has side effects (like os.makedirs).
        # So, we'll simulate the part of main that constructs the query.
        
        project_id = "TCGA-TEST"
        files_endpoint = "https://api.gdc.cancer.gov/files"
        
        # Expected filters
        expected_filters = {
            "op": "and",
            "content": [
                {"op": "in", "content": {"field": "cases.project.project_id", "value": [project_id]}},
                {"op": "in", "content": {"field": "access", "value": ["open"]}},
                {"op": "in", "content": {"field": "data_category", "value": ["Somatic Mutation"]}},
                {"op": "in", "content": {"field": "data_type", "value": ["Masked Somatic Mutation"]}},
                {"op": "in", "content": {"field": "experimental_strategy", "value": ["WXS"]}}
            ]
        }
        
        expected_params = {
            "filters": json.dumps(expected_filters),
            "fields": "file_id,file_name",
            "format": "JSON",
            "size": 100, # Default page size in script
            "from": 0
        }

        # This is a simplified call, assuming download_tcga_mafs.main() would make such a call.
        # To properly test, we'd need to refactor download_tcga_mafs to make the query logic
        # more accessible or capture the arguments to requests.get.
        
        # Let's run the relevant part of the script logic by calling main with mocked os.makedirs
        with mock.patch('os.makedirs'), \
             mock.patch('scripts.download_tcga_mafs.download_file'): # also mock download_file
            
            # Use a context manager to temporarily modify sys.argv for this specific test run
            with mock.patch('sys.argv', ['download_tcga_mafs.py', '--project_ids', project_id, '--output_dir', './fake_dir']):
                 download_tcga_mafs.main()


        # Assert that requests.get was called with the correct URL and parameters
        mock_requests_get.assert_called_with(files_endpoint, params=mock.ANY)
        actual_call_args = mock_requests_get.call_args
        actual_params_json = actual_call_args[1]['params'] # Get the params dict

        self.assertEqual(actual_params_json['fields'], expected_params['fields'])
        self.assertEqual(actual_params_json['format'], expected_params['format'])
        self.assertEqual(actual_params_json['size'], expected_params['size'])
        self.assertEqual(json.loads(actual_params_json['filters']), expected_filters)


    @mock.patch('os.makedirs')
    def test_directory_creation_logic(self, mock_os_makedirs):
        """Test that os.makedirs is called to create project-specific directories."""
        # Simulate running the script for a project
        # We need to mock requests.get to prevent actual API calls
        with mock.patch('scripts.download_tcga_mafs.requests.get') as mock_requests_get:
            # Mock a response that indicates no files, to quickly get past the API call logic
            mock_response = mock.Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "data": {"hits": [], "pagination": {"total": 0}}
            }
            mock_requests_get.return_value = mock_response

            test_project_id = 'TCGA-BRCA'
            test_output_dir = './test_output'
            expected_path = os.path.join(test_output_dir, test_project_id)

            # Use a context manager to temporarily modify sys.argv for this specific test run
            with mock.patch('sys.argv', ['download_tcga_mafs.py', '--project_ids', test_project_id, '--output_dir', test_output_dir]):
                download_tcga_mafs.main()
        
            mock_os_makedirs.assert_called_with(expected_path, exist_ok=True)

    @mock.patch('scripts.download_tcga_mafs.download_file') # Mock our own download_file function
    @mock.patch('scripts.download_tcga_mafs.requests.get') # Mock requests.get for metadata
    @mock.patch('os.makedirs') # Mock os.makedirs
    def test_file_download_and_naming(self, mock_os_makedirs, mock_metadata_get, mock_download_file_func):
        """Test file download attempt with correct naming and path."""
        project_id = 'TCGA-LUSC'
        output_dir = './test_data_download'
        file_id = 'some-fake-uuid-12345'
        file_name = 'TCGA.LUSC.mutect.somatic.maf.gz'
        expected_file_path = os.path.join(output_dir, project_id, file_name)
        data_download_url = f"https://api.gdc.cancer.gov/data/{file_id}"

        # Mock metadata API response
        mock_metadata_response = mock.Mock()
        mock_metadata_response.status_code = 200
        mock_metadata_response.json.return_value = {
            "data": {
                "hits": [{"file_id": file_id, "file_name": file_name}],
                "pagination": {"total": 1, "count": 1, "page": 1, "pages": 1, "from": 0, "size": 100}
            }
        }
        mock_metadata_get.return_value = mock_metadata_response

        # Simulate running main
        with mock.patch('sys.argv', ['download_tcga_mafs.py', '--project_ids', project_id, '--output_dir', output_dir]):
            download_tcga_mafs.main()

        # Check that os.makedirs was called for the project directory
        mock_os_makedirs.assert_any_call(os.path.join(output_dir, project_id), exist_ok=True)

        # Check that our download_file function was called with the correct URL and target path
        mock_download_file_func.assert_called_with(data_download_url, expected_file_path)


    @mock.patch('scripts.download_tcga_mafs.requests.get')
    @mock.patch('os.makedirs')
    @mock.patch('scripts.download_tcga_mafs.download_file')
    def test_api_pagination_logic(self, mock_download_file, mock_os_makedirs, mock_requests_get):
        """Test that the script correctly handles API pagination for metadata."""
        project_id = 'TCGA-GBM'
        output_dir = './test_pagination'

        # Simulate two pages of results
        mock_response_page1 = mock.Mock()
        mock_response_page1.status_code = 200
        mock_response_page1.json.return_value = {
            "data": {
                "hits": [
                    {"file_id": "uuid1", "file_name": "file1.maf.gz"},
                    {"file_id": "uuid2", "file_name": "file2.maf.gz"}
                ],
                "pagination": {"total": 3, "count": 2, "page": 1, "pages": 2, "from": 0, "size": 2}
            }
        }

        mock_response_page2 = mock.Mock()
        mock_response_page2.status_code = 200
        mock_response_page2.json.return_value = {
            "data": {
                "hits": [
                    {"file_id": "uuid3", "file_name": "file3.maf.gz"}
                ],
                "pagination": {"total": 3, "count": 1, "page": 2, "pages": 2, "from": 2, "size": 2}
            }
        }
        
        # Set up the mock to return different responses on subsequent calls
        mock_requests_get.side_effect = [mock_response_page1, mock_response_page2]

        with mock.patch('sys.argv', ['download_tcga_mafs.py', '--project_ids', project_id, '--output_dir', output_dir]):
            download_tcga_mafs.main()

        # Check that requests.get was called twice for the files endpoint
        files_endpoint_url = "https://api.gdc.cancer.gov/files"
        
        # Get all calls to requests.get
        calls_to_get = [call for call in mock_requests_get.call_args_list if call[0][0] == files_endpoint_url]
        self.assertEqual(len(calls_to_get), 2)

        # Verify 'from' parameter in calls
        self.assertEqual(json.loads(calls_to_get[0][1]['params']['filters'])['content'][0]['content']['value'], [project_id])
        self.assertEqual(calls_to_get[0][1]['params']['from'], 0)
        self.assertEqual(calls_to_get[0][1]['params']['size'], 2) # Based on mock response

        self.assertEqual(json.loads(calls_to_get[1][1]['params']['filters'])['content'][0]['content']['value'], [project_id])
        self.assertEqual(calls_to_get[1][1]['params']['from'], 2) # from should be 0 + 2
        self.assertEqual(calls_to_get[1][1]['params']['size'], 2)


        # And that download_file was called for all 3 files
        self.assertEqual(mock_download_file.call_count, 3)
        mock_download_file.assert_any_call(f"https://api.gdc.cancer.gov/data/uuid1", os.path.join(output_dir, project_id, "file1.maf.gz"))
        mock_download_file.assert_any_call(f"https://api.gdc.cancer.gov/data/uuid2", os.path.join(output_dir, project_id, "file2.maf.gz"))
        mock_download_file.assert_any_call(f"https://api.gdc.cancer.gov/data/uuid3", os.path.join(output_dir, project_id, "file3.maf.gz"))


    @mock.patch('scripts.download_tcga_mafs.requests.get')
    @mock.patch('os.makedirs')
    @mock.patch('builtins.print') # To check error messages
    def test_error_handling_metadata_api_request_exception(self, mock_print_builtin, mock_os_makedirs, mock_requests_get):
        """Test error handling when metadata API call raises RequestException."""
        project_id = 'TCGA-ERR'
        output_dir = './test_error'
        
        mock_requests_get.side_effect = requests.exceptions.RequestException("Test network error")

        with mock.patch('sys.argv', ['download_tcga_mafs.py', '--project_ids', project_id, '--output_dir', output_dir]):
            download_tcga_mafs.main()
        
        mock_print_builtin.assert_any_call(f"Error fetching file list for {project_id}: Test network error")

    @mock.patch('scripts.download_tcga_mafs.requests.get')
    @mock.patch('os.makedirs')
    @mock.patch('builtins.print') # To check error messages
    def test_error_handling_metadata_api_http_error(self, mock_print_builtin, mock_os_makedirs, mock_requests_get):
        """Test error handling when metadata API call returns non-200 status."""
        project_id = 'TCGA-HTTPERR'
        output_dir = './test_httperror'

        mock_response = mock.Mock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("Server error")
        mock_requests_get.return_value = mock_response
        
        with mock.patch('sys.argv', ['download_tcga_mafs.py', '--project_ids', project_id, '--output_dir', output_dir]):
            download_tcga_mafs.main()

        # The script's error message for HTTPError comes from the exception itself
        mock_print_builtin.assert_any_call(f"Error fetching file list for {project_id}: Server error")


    @mock.patch('scripts.download_tcga_mafs.requests.get') # Mocks requests.get for the main script
    @mock.patch('os.makedirs')
    @mock.patch('builtins.print') # To check error messages from download_file
    def test_error_handling_file_download_request_exception(self, mock_print_builtin, mock_os_makedirs, mock_main_requests_get):
        """Test error handling when file download API call raises RequestException."""
        project_id = 'TCGA-DL-ERR'
        output_dir = './test_dl_error'
        file_id = 'err-uuid-1'
        file_name = 'error_file.maf.gz'
        download_url = f"https://api.gdc.cancer.gov/data/{file_id}"
        target_file_path = os.path.join(output_dir, project_id, file_name)

        # Mock metadata API response to provide one file
        mock_metadata_response = mock.Mock()
        mock_metadata_response.status_code = 200
        mock_metadata_response.json.return_value = {
            "data": {
                "hits": [{"file_id": file_id, "file_name": file_name}],
                "pagination": {"total": 1, "count": 1, "page": 1, "pages": 1, "from": 0, "size": 100}
            }
        }
        
        # Mock for the download_file's requests.get call
        # This requires careful patching as download_file is in the same module.
        # We'll make the main metadata call succeed, then make the download_file's call fail.
        
        # Patch 'requests.get' specifically in the scope of 'download_tcga_mafs.download_file'
        with mock.patch('scripts.download_tcga_mafs.requests.get', side_effect=[
            mock_metadata_response, # First call (metadata) succeeds
            requests.exceptions.RequestException("Download network error") # Second call (download) fails
        ]) as mock_requests_get_for_download:
            with mock.patch('sys.argv', ['download_tcga_mafs.py', '--project_ids', project_id, '--output_dir', output_dir]):
                download_tcga_mafs.main()

        # Check that the error print from within download_file was called
        mock_print_builtin.assert_any_call(f"Error downloading {download_url}: Download network error")


    @mock.patch('scripts.download_tcga_mafs.requests.get') # Mocks requests.get for the main script
    @mock.patch('os.makedirs')
    @mock.patch('builtins.print') # To check error messages from download_file
    def test_error_handling_file_download_http_error(self, mock_print_builtin, mock_os_makedirs, mock_main_requests_get):
        """Test error handling when file download API call returns non-200 status."""
        project_id = 'TCGA-DL-HTTPERR'
        output_dir = './test_dl_httperror'
        file_id = 'err-uuid-2'
        file_name = 'error_file_http.maf.gz'
        download_url = f"https://api.gdc.cancer.gov/data/{file_id}"

        # Mock metadata API response
        mock_metadata_response = mock.Mock()
        mock_metadata_response.status_code = 200
        mock_metadata_response.json.return_value = {
            "data": {
                "hits": [{"file_id": file_id, "file_name": file_name}],
                "pagination": {"total": 1, "count": 1, "page": 1, "pages": 1, "from": 0, "size": 100}
            }
        }

        # Mock for the download_file's requests.get call (HTTP error)
        mock_download_error_response = mock.Mock()
        mock_download_error_response.status_code = 404
        mock_download_error_response.raise_for_status.side_effect = requests.exceptions.HTTPError("File not found")
        
        # Patch 'requests.get' to handle both metadata and download calls
        with mock.patch('scripts.download_tcga_mafs.requests.get', side_effect=[
            mock_metadata_response,         # Metadata call
            mock_download_error_response    # Download call
        ]) as mock_requests_get_for_download:
             with mock.patch('sys.argv', ['download_tcga_mafs.py', '--project_ids', project_id, '--output_dir', output_dir]):
                download_tcga_mafs.main()
        
        mock_print_builtin.assert_any_call(f"Error downloading {download_url}: File not found")

if __name__ == '__main__':
    unittest.main()
