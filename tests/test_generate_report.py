import unittest
from unittest.mock import patch, mock_open, MagicMock
import argparse
import os
import sys
import io # For capturing stderr
import datetime # For checking date in title page

# Adjust sys.path to allow direct import of the script under test
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import functions from the script to be tested
from scripts import generate_report # Import the module itself to access its functions

class TestGenerateReport(unittest.TestCase):
    """Test suite for generate_report.py script."""

    def test_argument_parser(self):
        """Tests the argument parsing functionality."""
        parser = generate_report.create_parser()

        output_file_value = 'my_report.pdf'
        args_list_valid = ['--output_pdf_file', output_file_value]
        args = parser.parse_args(args_list_valid)
        self.assertEqual(args.output_pdf_file, output_file_value)

        args_list_empty = []
        with self.assertRaises(SystemExit, msg="Should exit when --output_pdf_file is missing"):
            parser.parse_args(args_list_empty)

    # --- Tests for HTML Text Section Functions ---
    def test_generate_title_page_html_content(self):
        """Tests the content of the title page HTML."""
        result = generate_report.generate_title_page_html()
        self.assertIn("<h1>Pan-Cancer Mutational Signature Discovery (k=5) and Downstream Analysis</h1>", result)
        self.assertIn("<p class=\"title-page-author\">Author: SnackitPal</p>", result)
        current_date_str = datetime.date.today().strftime("%Y-%m-%d")
        self.assertIn(f"<p class=\"title-page-date\">Date: {current_date_str}</p>", result)

    def test_generate_abstract_html_content(self):
        """Tests the content of the abstract HTML."""
        result = generate_report.generate_abstract_html()
        self.assertIn("<h2>Abstract</h2>", result)
        self.assertIn("summarizes the discovery of k=5 mutational signatures", result)
    
    def test_generate_methods_html_content(self):
        """Tests content of methods HTML for a keyword."""
        result = generate_report.generate_methods_html()
        self.assertIn("<h2>Methods</h2>", result)
        self.assertIn("Latent Dirichlet Allocation (LDA)", result)


    # --- Test encode_image_to_base64 ---
    @patch('builtins.open', new_callable=mock_open, read_data=b'fake_image_data_bytes')
    def test_encode_image_to_base64_success(self, mock_file):
        """Tests successful base64 encoding of an image."""
        result = generate_report.encode_image_to_base64('dummy_path.png')
        self.assertTrue(result.startswith('data:image/png;base64,'))
        # Check if the encoded part is correct for 'fake_image_data_bytes'
        # b'fake_image_data_bytes' -> 'ZmFrZV9pbWFnZV9kYXRhX2J5dGVz'
        self.assertIn('ZmFrZV9pbWFnZV9kYXRhX2J5dGVz', result)

    @patch('builtins.open', side_effect=FileNotFoundError("File not found for test"))
    @patch('sys.stderr', new_callable=io.StringIO)
    def test_encode_image_to_base64_file_not_found(self, mock_stderr, mock_open_file):
        """Tests behavior when image file is not found for base64 encoding."""
        result = generate_report.encode_image_to_base64('nonexistent.png')
        self.assertIsNone(result)
        self.assertIn("Error: Image not found at nonexistent.png", mock_stderr.getvalue())

    # --- Test generate_figure_html ---
    @patch('scripts.generate_report.encode_image_to_base64', return_value='fake_base64_image_data_uri_string')
    def test_generate_figure_html_image_found(self, mock_encode_image):
        """Tests figure HTML generation when the image is found and encoded."""
        result = generate_report.generate_figure_html('dummy/path.png', 'Test Caption Text', 7)
        mock_encode_image.assert_called_once_with('dummy/path.png')
        self.assertIn('<img src="fake_base64_image_data_uri_string" alt="Test Caption Text">', result)
        self.assertIn('<p class="caption">Figure 7: Test Caption Text</p>', result)
        self.assertNotIn('style="color: red;"', result) # No error style

    @patch('scripts.generate_report.encode_image_to_base64', return_value=None)
    def test_generate_figure_html_image_not_found(self, mock_encode_image):
        """Tests figure HTML generation when the image is not found (encode returns None)."""
        image_path = 'path/that/fails.png'
        result = generate_report.generate_figure_html(image_path, 'Failed Caption', 8)
        mock_encode_image.assert_called_once_with(image_path)
        self.assertNotIn('<img src=', result) # No img tag src
        self.assertIn(f'Figure 8: Failed Caption (Error: Image not found at {image_path})</p>', result)
        self.assertIn('style="color: red;"', result) # Error style should be present

    # --- Test Main PDF Generation Logic (generate_report.main) with Mocks ---
    @patch('scripts.generate_report.HTML') # Mock the WeasyPrint HTML class constructor
    @patch('os.makedirs') # Mock os.makedirs
    @patch('scripts.generate_report.generate_report_content', return_value='<html><body>Mocked HTML Content</body></html>')
    def test_main_function_calls(self, mock_generate_content, mock_os_makedirs, mock_HTML_class):
        """Tests the main function's orchestration of calls with mocks."""
        
        mock_html_instance = MagicMock() # This will be the instance returned by HTML()
        mock_HTML_class.return_value = mock_html_instance # HTML() will return our mock_html_instance

        output_pdf_path = 'test_output_dir/final_report.pdf'
        
        # Use a context manager to temporarily modify sys.argv if needed, or pass args_list directly
        # For simplicity, directly calling main with args_list
        generate_report.main(['--output_pdf_file', output_pdf_path])

        # Assert os.makedirs was called to create the directory
        mock_os_makedirs.assert_called_once_with(os.path.dirname(output_pdf_path), exist_ok=True)
        
        # Assert generate_report_content was called
        mock_generate_content.assert_called_once()
        # args passed to generate_report_content will be an argparse.Namespace object
        call_args_for_content = mock_generate_content.call_args[0][0]
        self.assertEqual(call_args_for_content.output_pdf_file, output_pdf_path)
        
        # Assert HTML class was instantiated correctly
        mock_HTML_class.assert_called_once_with(
            string='<html><body>Mocked HTML Content</body></html>',
            base_url=os.getcwd()
        )
        
        # Assert write_pdf was called on the HTML instance
        # The font_config might be a new instance each time, so we check it was called with some FontConfiguration
        mock_html_instance.write_pdf.assert_called_once()
        call_args_for_write_pdf = mock_html_instance.write_pdf.call_args
        self.assertEqual(call_args_for_write_pdf[0][0], output_pdf_path) # First positional arg
        self.assertIsInstance(call_args_for_write_pdf[1]['font_config'], generate_report.FontConfiguration)


    # --- Test generate_report_content (Overall Structure Check) ---
    @patch('scripts.generate_report.generate_figure_html', return_value="<p>--Mocked Figure HTML--</p>")
    def test_generate_report_content_structure(self, mock_gen_figure_html):
        """Tests the overall structure and calls within generate_report_content."""
        # Create a mock 'args' object; generate_report_content doesn't use args in current form,
        # but it's good practice to pass something similar to what it expects.
        mock_args = MagicMock() 
        
        html_content = generate_report.generate_report_content(mock_args)

        # Check for overall HTML structure
        self.assertTrue(html_content.startswith("<!DOCTYPE html>"))
        self.assertIn("<title>Analysis Report</title>", html_content)
        self.assertIn("<style>", html_content)
        self.assertIn(generate_report.DEFAULT_CSS, html_content) # Check if CSS is included
        self.assertIn("<body>", html_content)
        self.assertTrue(html_content.endswith("</html>"))

        # Check for presence of key sections
        self.assertIn("<h1>Pan-Cancer Mutational Signature Discovery", html_content) # Title page
        self.assertIn("<h2>Abstract</h2>", html_content)
        self.assertIn("<h2>Introduction</h2>", html_content)
        self.assertIn("<h2>Methods</h2>", html_content)
        self.assertIn("<h2>Results</h2>", html_content)
        self.assertIn("<h3>5.1. TCGA Data Acquisition", html_content)
        self.assertIn("<h4>5.5.1. LUAD / Smoking Signature", html_content)
        self.assertIn("<h2>Discussion</h2>", html_content)
        self.assertIn("<h2>Conclusion</h2>", html_content)
        self.assertIn("<h2>References</h2>", html_content)

        # Assert that generate_figure_html was called the expected number of times
        # 5 LDA sigs + 1 COSMIC heatmap + 1 Avg Exp + 5 Exp Boxplots + 3 GSEA plots = 15 figures
        self.assertEqual(mock_gen_figure_html.call_count, 15)
        
        # Check one example call to generate_figure_html to ensure paths are being formed correctly (optional)
        # This requires knowing the exact call order or using more sophisticated mocking.
        # For simplicity, call_count is a good indicator that the logic is iterating as expected.
        # Example of checking a specific call (if calls are consistently ordered):
        # first_figure_call_args = mock_gen_figure_html.call_args_list[0][0] # (image_path, caption, number)
        # self.assertEqual(first_figure_call_args[0], "./results/figures/signatures/lda_k5_s42_Signature_1.png")


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
