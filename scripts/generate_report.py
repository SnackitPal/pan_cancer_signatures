"""
Generates a PDF report from various analysis results using WeasyPrint.

This script will eventually combine text, tables, and plots into a single
PDF document.
"""
import argparse
import os
import sys
import datetime
import base64 # Added for image embedding
from weasyprint import HTML, CSS
from weasyprint.fonts import FontConfiguration # For potential advanced font control

DEFAULT_CSS = """
    @page {
        margin: 0.75in; /* Standard page margin */
    }
    body {
        font-family: sans-serif;
        /* Body margin is often less critical if @page margin is set, 
           but can be useful for content flow within the page box.
           Setting to 0 here as @page handles the main margins. */
        margin: 0; 
        line-height: 1.5;
        font-size: 10pt;
    }
    h1, h2, h3, h4 { /* Added h4 */
        margin-top: 1.5em;
        margin-bottom: 0.5em;
        font-weight: bold;
    }
    h1 { font-size: 24pt; text-align: center; page-break-after: always; } /* Title page specific */
    h2 { font-size: 18pt; page-break-before: always; } /* New major sections on new page */
    h3 { font-size: 14pt; }
    h4 { font-size: 12pt; } /* For sub-sub-sections like GSEA results */
    p {
        margin-bottom: 0.75em;
    }
    .title-page-author, .title-page-date { /* Specific styles for title page author/date */
        text-align: center;
        font-size: 12pt;
        margin-bottom: 0.5em;
    }
    .figure {
        margin-top: 1.5em;
        margin-bottom: 1.5em;
        text-align: center;
        page-break-inside: avoid;
    }
    .figure img {
        max-width: 90%; 
        height: auto;
        display: block;
        margin-left: auto;
        margin-right: auto;
        border: 1px solid #ccc; 
    }
    .caption {
        font-size: 0.9em;
        font-style: italic;
        margin-top: 0.5em;
        text-align: center;
    }
    table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 1em;
        margin-bottom: 1em;
        page-break-inside: avoid;
    }
    th, td {
        border: 1px solid #ddd;
        padding: 8px;
        text-align: left;
    }
    th {
        background-color: #f2f2f2;
    }
"""

PAGE_TEMPLATE_HTML = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Analysis Report</title>
    <style>
        {css_styles}
    </style>
</head>
<body>
    {body_content}
</body>
</html>
"""

def create_parser():
    """Creates and returns the ArgumentParser object for the script."""
    parser = argparse.ArgumentParser(description="Generate a PDF report from analysis results.")
    parser.add_argument(
        "--output_pdf_file",
        type=str,
        required=True,
        help="Path to save the output PDF report (e.g., './reports/Analysis_Report.pdf')."
    )
    # Add other arguments here as needed, e.g., paths to input files for the report
    return parser

# --- Image Encoding and Figure HTML Generation ---
def encode_image_to_base64(image_path):
    """
    Encodes an image file to a base64 data URI.
    Assumes PNG format.
    """
    try:
        with open(image_path, 'rb') as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return f"data:image/png;base64,{encoded_string}"
    except FileNotFoundError:
        print(f"Error: Image not found at {image_path}", file=sys.stderr)
        return None

def generate_figure_html(image_path, caption_text, figure_number):
    """
    Generates HTML for embedding a figure with a caption.
    Handles cases where the image might not be found.
    """
    base64_image_data = encode_image_to_base64(image_path)
    
    if base64_image_data:
        return f"""
        <div class="figure">
            <img src="{base64_image_data}" alt="{caption_text}">
            <p class="caption">Figure {figure_number}: {caption_text}</p>
        </div>
        """
    else:
        return f"""
        <div class="figure">
            <p class="caption" style="color: red;">Figure {figure_number}: {caption_text} (Error: Image not found at {image_path})</p>
        </div>
        """

# --- Section-Specific HTML Generation Functions (Text Only) ---

def generate_title_page_html():
    """Generates HTML for the title page."""
    current_date = datetime.date.today().strftime("%Y-%m-%d")
    return f"""
        <h1>Pan-Cancer Mutational Signature Discovery (k=5) and Downstream Analysis</h1>
        <p class="title-page-author">Author: SnackitPal</p>
        <p class="title-page-date">Date: {current_date}</p>
    """

def generate_abstract_html():
    """Generates HTML for the Abstract section."""
    return """
        <h2>Abstract</h2>
        <p>This report summarizes the discovery of k=5 mutational signatures from pan-cancer whole-exome sequencing data, their comparison to known COSMIC signatures, and subsequent downstream analyses. These analyses include cohort-specific signature exposures and Gene Set Enrichment Analysis (GSEA) to identify biological pathways associated with high exposures to specific signatures in selected TCGA cohorts (LUAD, SKCM, BRCA).</p>
    """

def generate_introduction_html():
    """Generates HTML for the Introduction section."""
    return """
        <h2>Introduction</h2>
        <p>Mutational signatures are characteristic patterns of somatic mutations that arise from distinct mutagenic processes. Understanding these signatures can provide insights into cancer etiology, tumor evolution, and potential therapeutic vulnerabilities. This project aims to perform de novo discovery of mutational signatures from a pan-cancer dataset, characterize these signatures, and explore their biological relevance through downstream analyses.</p>
    """

def generate_methods_html():
    """Generates HTML for the Methods section."""
    return """
        <h2>Methods</h2>
        <p>The methodology involved several stages: (1) Data acquisition and preprocessing of TCGA WES MAF files for selected cohorts. (2) Generation of a sample-by-mutation type matrix. (3) Application of Latent Dirichlet Allocation (LDA) for de novo signature discovery (k=5). (4) Comparison of discovered signatures to the COSMIC database. (5) Analysis of signature exposures in individual patients and across cohorts. (6) Gene Set Enrichment Analysis (GSEA) for pathways associated with high signature exposures in LUAD (Lung Adenocarcinoma), SKCM (Skin Cutaneous Melanoma), and BRCA (Breast Cancer) cohorts, focusing on Hallmark gene sets.</p>
    """

def generate_results_tcga_summary_text_html():
    """Generates HTML for the TCGA Data Summary section (text only)."""
    return """
        <h3>5.1. TCGA Data Acquisition & Preprocessing Summary</h3>
        <p>Placeholder text summarizing cohorts LUAD, SKCM, BRCA. Data was downloaded from GDC, filtered, and processed into a mutation count matrix suitable for LDA analysis.</p>
    """

def generate_results_lda_intro_text_html():
    """Generates HTML for the LDA Modeling Introduction section (text only)."""
    return """
        <h3>5.2. LDA Modeling: k=5 Mutational Signature Discovery</h3>
        <p>Placeholder text introducing the 5 discovered signatures. Details on model selection (k=5) and signature profiles will be presented.</p>
    """

def generate_results_cosmic_intro_text_html():
    """Generates HTML for the COSMIC Comparison Introduction section (text only)."""
    return """
        <h3>5.3. Comparison to COSMIC Signatures</h3>
        <p>Placeholder text summarizing the comparison of the 5 discovered signatures to known COSMIC v3 signatures. Cosine similarity scores and potential matches will be discussed.</p>
    """

def generate_results_exposure_intro_text_html():
    """Generates HTML for the Patient Signature Exposure Introduction section (text only)."""
    return """
        <h3>5.4. Patient Signature Exposure Analysis (k=5)</h3>
        <p>Placeholder text summarizing cohort-specific exposures. Distribution of signature exposures across LUAD, SKCM, and BRCA cohorts will be visualized.</p>
    """

def generate_results_gsea_intro_text_html():
    """Generates HTML for the GSEA Introduction section (text only)."""
    return """
        <h3>5.5. Gene Set Enrichment Analysis (GSEA) - Hallmark Gene Sets</h3>
        <p>Placeholder text introducing GSEA results. GSEA was performed on ranked gene lists derived from differential mutation burden between high and low exposure groups for selected signatures in specific cohorts.</p>
    """

def generate_results_gsea_luad_text_html():
    """Generates HTML for the LUAD GSEA results section (text only)."""
    return """
        <h4>5.5.1. LUAD / Smoking Signature (Signature_3)</h4>
        <p>(Placeholder for text describing GSEA results for LUAD and the presumed smoking signature.)</p>
    """

def generate_results_gsea_skcm_text_html():
    """Generates HTML for the SKCM GSEA results section (text only)."""
    return """
        <h4>5.5.2. SKCM / UV Signature (Signature_2)</h4>
        <p>(Placeholder for text describing GSEA results for SKCM and the presumed UV signature.)</p>
    """

def generate_results_gsea_brca_text_html():
    """Generates HTML for the BRCA GSEA results section (text only)."""
    return """
        <h4>5.5.3. BRCA / APOBEC Signature (Signature_4)</h4>
        <p>(Placeholder for text describing GSEA results for BRCA and the presumed APOBEC signature.)</p>
    """

def generate_discussion_html():
    """Generates HTML for the Discussion section."""
    return """
        <h2>Discussion</h2>
        <p>Placeholder for discussion of results, limitations, and potential future work. This section will interpret the findings from signature discovery, COSMIC comparison, exposure analysis, and GSEA, integrating them into a cohesive narrative.</p>
    """

def generate_conclusion_html():
    """Generates HTML for the Conclusion section."""
    return """
        <h2>Conclusion</h2>
        <p>Placeholder for concise summary of key findings and their implications. This project successfully discovered and characterized k=5 mutational signatures and explored their biological significance through GSEA.</p>
    """

def generate_references_html():
    """Generates HTML for the References section."""
    return """
        <h2>References</h2>
        <p>Placeholder for citations. (e.g., Alexandrov et al., COSMIC database, GSEA publications, WeasyPrint, etc.)</p>
    """

# --- Main Report Content Generation ---
def generate_report_content(args):
    """
    Generates the HTML content for the report by assembling parts including figures.
    """
    print("Generating report content sections...")
    html_parts = []
    current_figure_number = 1

    # --- Title Page ---
    html_parts.append(generate_title_page_html())
    # --- Abstract ---
    html_parts.append(generate_abstract_html())
    # --- Introduction ---
    html_parts.append(generate_introduction_html())
    # --- Methods ---
    html_parts.append(generate_methods_html())
    
    # --- Results ---
    html_parts.append("<h2>Results</h2>") # Main Results Heading
    html_parts.append(generate_results_tcga_summary_text_html())
    
    html_parts.append(generate_results_lda_intro_text_html())
    for i in range(1, 6): # 5 LDA signatures
        lda_sig_path = f"./results/figures/signatures/lda_k5_s42_Signature_{i}.png"
        lda_sig_caption = f"Profile of Discovered Signature {i}"
        html_parts.append(generate_figure_html(lda_sig_path, lda_sig_caption, current_figure_number))
        current_figure_number += 1
        
    html_parts.append(generate_results_cosmic_intro_text_html())
    cosmic_heatmap_path = "./results/figures/comparison/cosine_heatmap_k5_s42.png"
    cosmic_heatmap_caption = "Cosine Similarity of Discovered Signatures to COSMIC SBS Signatures"
    html_parts.append(generate_figure_html(cosmic_heatmap_path, cosmic_heatmap_caption, current_figure_number))
    current_figure_number += 1
    
    html_parts.append(generate_results_exposure_intro_text_html())
    avg_contrib_path = "./results/figures/exposures/avg_signature_contributions_by_cohort.png"
    avg_contrib_caption = "Average Signature Contributions by Cohort"
    html_parts.append(generate_figure_html(avg_contrib_path, avg_contrib_caption, current_figure_number))
    current_figure_number += 1
    for i in range(1, 6): # 5 exposure boxplots
        exp_boxplot_path = f"./results/figures/exposures/exposure_boxplot_Signature_{i}.png"
        exp_boxplot_caption = f"Exposure Boxplot for Signature {i}"
        html_parts.append(generate_figure_html(exp_boxplot_path, exp_boxplot_caption, current_figure_number))
        current_figure_number += 1

    html_parts.append(generate_results_gsea_intro_text_html())
    
    html_parts.append(generate_results_gsea_luad_text_html())
    luad_gsea_path = "./results/figures/gsea_summary/LUAD_Signature_3_top20_pathways.png"
    luad_gsea_caption = "GSEA Hallmark Pathway Summary for LUAD / Signature 3 (e.g., Top Hit: Epithelial Mesenchymal Transition)"
    html_parts.append(generate_figure_html(luad_gsea_path, luad_gsea_caption, current_figure_number))
    current_figure_number += 1

    html_parts.append(generate_results_gsea_skcm_text_html())
    skcm_gsea_path = "./results/figures/gsea_summary/SKCM_Signature_2_top20_pathways.png"
    skcm_gsea_caption = "GSEA Hallmark Pathway Summary for SKCM / Signature 2"
    html_parts.append(generate_figure_html(skcm_gsea_path, skcm_gsea_caption, current_figure_number))
    current_figure_number += 1

    html_parts.append(generate_results_gsea_brca_text_html())
    brca_gsea_path = "./results/figures/gsea_summary/BRCA_Signature_4_top20_pathways.png"
    brca_gsea_caption = "GSEA Hallmark Pathway Summary for BRCA / Signature 4"
    html_parts.append(generate_figure_html(brca_gsea_path, brca_gsea_caption, current_figure_number))
    current_figure_number += 1
    
    # --- Discussion ---
    html_parts.append(generate_discussion_html())
    # --- Conclusion ---
    html_parts.append(generate_conclusion_html())
    # --- References ---
    html_parts.append(generate_references_html())

    body_content = "\n".join(html_parts)
    return PAGE_TEMPLATE_HTML.format(css_styles=DEFAULT_CSS, body_content=body_content)

def main(args_list=None):
    """
    Main function to parse arguments, generate report content, and create the PDF.
    """
    parser = create_parser()
    args = parser.parse_args(args_list if args_list is not None else sys.argv[1:])

    print("Starting PDF report generation...")
    print(f"Output PDF will be saved to: {args.output_pdf_file}")

    # Create output directory if it doesn't exist and if a directory path is specified
    output_dir = os.path.dirname(args.output_pdf_file)
    if output_dir: # Only try to create if output_dir is not an empty string (i.e., path has a directory part)
        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir, exist_ok=True)
                print(f"Created output directory: {output_dir}")
            except Exception as e:
                print(f"Error: Could not create output directory {output_dir}. {e}", file=sys.stderr)
                sys.exit(1)
    else:
        # If output_dir is empty, PDF will be created in the current working directory.
        print("Outputting PDF to the current directory.")

    print("Generating HTML content...")
    html_content = generate_report_content(args)
    print("Generated HTML content.")

    print("Initializing WeasyPrint to render PDF...")
    try:
        font_config = FontConfiguration()
        # base_url=os.getcwd() is good practice for resolving any relative paths if they were used
        # (e.g., for external CSS not embedded, or if images weren't base64 encoded).
        html_doc = HTML(string=html_content, base_url=os.getcwd()) 
        
        print(f"Writing PDF to {args.output_pdf_file}...")
        html_doc.write_pdf(args.output_pdf_file, font_config=font_config)
        
        print(f"PDF report generated successfully: {args.output_pdf_file}")

    except Exception as e:
        print(f"Error during PDF generation with WeasyPrint for '{args.output_pdf_file}': {e}", file=sys.stderr)
        print("This error can occur if WeasyPrint or its system dependencies (like Pango, Cairo, GDK-PixBuf) "
              "are not correctly installed or configured in your environment.", file=sys.stderr)
        print("If running in a CI/CD pipeline or container, ensure these system libraries are included in the image/setup.", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
