# Pan-Cancer Mutational Signature Discovery and Downstream Analysis

## Project Overview

This project aims to perform de novo discovery of mutational signatures from pan-cancer whole-exome sequencing (WES) data obtained from The Cancer Genome Atlas (TCGA). It utilizes probabilistic topic models (Latent Dirichlet Allocation - LDA) to identify signatures, compares them to known COSMIC signatures, analyzes patient-specific signature exposures across different cancer types, and performs Gene Set Enrichment Analysis (GSEA) to uncover associated biological pathways.

The analytical pipeline is scripted primarily in Python, with significant portions of the code generated and refined with the assistance of "Jules," an AI coding agent.

**Initial Analysis (k=5 Signatures):**
*   Cohorts: TCGA-LUAD (Lung Adenocarcinoma), TCGA-SKCM (Skin Cutaneous Melanoma), TCGA-BRCA (Breast Invasive Carcinoma).
*   Key findings included identification of UV, Smoking, APOBEC, and MMRd/Aging-like signatures and their GSEA pathway associations.

**Extended Analysis (k=7 Signatures):**
*   Cohorts: TCGA-UCEC (Uterine Corpus Endometrial Carcinoma), TCGA-HNSC (Head and Neck Squamous Cell Carcinoma), TCGA-STAD (Stomach Adenocarcinoma).
*   Key findings included identification of Aging, multiple MMRd types, POLE, APOBEC, and HRd-like signatures and their GSEA pathway associations.

## Repository Structure
pan_cancer_signatures_jules/
├── data/ # (Gitignored) Base directory for input data
│ ├── raw_mafs/ # Raw MAF files downloaded from GDC (organized by cohort)
│ ├── processed/ # Processed data (mutation catalogs, sample maps, TMB)
│ └── reference_data/ # Reference files (COSMIC signatures, GMT files)
│ └── msigdb/ # Gene set files (e.g., Hallmark GMT)
├── results/ # (Gitignored) Base directory for k=5 analysis results
│ ├── comparison/ # Cosine similarities (CSV) for k=5
│ ├── figures/ # Plots for k=5 analysis
│ │ ├── comparison/ # Heatmap for COSMIC comparison (k=5)
│ │ ├── exposures/ # Patient exposure plots (k=5)
│ │ ├── gsea_summary/ # GSEA summary plots (k=5)
│ │ └── signatures/ # Discovered signature profile plots (k=5)
│ ├── gsea_analysis/ # Full GSEApy output directories (k=5)
│ ├── gsea_input/ # Ranked gene lists (.rnk) for GSEA (k=5)
│ └── models/ # Trained LDA models, profiles, exposures (k=5)
├── data_k7_UCEC_HNSC_STAD/ # (Gitignored) Base for k=7 UCEC/HNSC/STAD data
│ ├── raw_mafs/
│ └── processed/
├── results_k7_UCEC_HNSC_STAD/ # (Gitignored) Base for k=7 UCEC/HNSC/STAD results
│ ├── comparison/
│ ├── figures/
│ │ ├── comparison/
│ │ ├── exposures_k7_UHS/
│ │ ├── gsea_summary/
│ │ └── signatures/
│ ├── gsea_analysis/
│ ├── gsea_input/
│ └── models/
├── reports/ # (Gitignored) Generated PDF reports
│ └── reports_k7_UCEC_HNSC_STAD/ # Specific reports for k=7 run
├── scripts/ # All Python scripts for the pipeline
│ ├── download_tcga_mafs.py # Downloads MAF files
│ ├── preprocess_mafs.py # MAF to mutation catalog
│ ├── train_lda_model.py # Trains LDA model
│ ├── signature_plotting_utils.py # Helper for signature plots
│ ├── visualize_signatures.py # Generates signature profile plots
│ ├── compare_to_cosmic.py # Compares signatures to COSMIC
│ ├── generate_sample_cohort_map.py # Creates sample-to-cohort mapping
│ ├── analyze_patient_exposures.py # Analyzes & plots signature exposures
│ ├── calculate_tmb.py # Calculates Tumor Mutation Burden
│ ├── generate_regression_ranked_list.py # Generates ranked gene list for GSEA (regression method)
│ ├── run_gsea_analysis.py # Runs GSEA using gseapy
│ ├── plot_gsea_summary.py # Plots GSEA summary results
│ └── generate_report.py # Generates final PDF report
├── tests/ # Unit tests for the scripts
│ ├── test_...py # Individual test files
├── venv/ # (Gitignored) Python virtual environment
├── .gitignore # Specifies intentionally untracked files
├── LICENSE # Project license (e.g., MIT)
├── main_pipeline.py # Master orchestration script for the pipeline
└── README.md # This file

## Key Pipeline Stages & Scripts

1.  **Data Acquisition (`scripts/download_tcga_mafs.py`):**
    *   Downloads open-access WXS (Whole Exome Sequencing) MAF (Mutation Annotation Format) files from the GDC API for specified TCGA project IDs.
    *   Key filters: `access: "open"`, `data_category: "Simple Nucleotide Variation"`, `data_type: "Masked Somatic Mutation"`, `experimental_strategy: "WXS"`, `data_format: "MAF"`.

2.  **MAF Preprocessing (`scripts/preprocess_mafs.py`):**
    *   Processes gzipped MAF files from cohort-specific subdirectories.
    *   Filters for Somatic Single Nucleotide Variants (SNVs).
    *   Normalizes chromosome names for FASTA compatibility.
    *   Uses `pyfaidx` with a reference genome (e.g., hg38.fa) to determine the 96 trinucleotide contexts for each SNV, including pyrimidine normalization.
    *   Aggregates mutation counts per sample into a [Samples x 96 Contexts] matrix.
    *   Output: `mutation_catalog_[...].csv`

3.  **LDA Model Training (`scripts/train_lda_model.py`):**
    *   Trains a scikit-learn LDA model on the mutation catalog.
    *   Takes `k` (number of signatures) and `random_seed` as parameters.
    *   Outputs:
        *   Saved trained LDA model (`.joblib`).
        *   Signature profiles ([k Signatures x 96 Contexts] probabilities).
        *   Patient exposures ([Samples x k Signatures] contributions).

4.  **Signature Visualization (`scripts/visualize_signatures.py` & `scripts/signature_plotting_utils.py`):**
    *   Generates standard 96-bar plots for each discovered signature profile.

5.  **COSMIC Comparison (`scripts/compare_to_cosmic.py`):**
    *   Calculates cosine similarities between discovered signature profiles and known COSMIC v3.4 SBS signatures.
    *   Outputs a CSV of similarities and a summary heatmap.

6.  **Sample-to-Cohort Mapping (`scripts/generate_sample_cohort_map.py`):**
    *   Creates a mapping file linking `Tumor_Sample_Barcode` to its cancer cohort based on MAF file directory structure.

7.  **Patient Exposure Analysis (`scripts/analyze_patient_exposures.py`):**
    *   Loads patient exposures and the sample-cohort map.
    *   Generates boxplots of individual signature exposures per cohort and a stacked bar plot of average signature contributions per cohort.

8.  **Tumor Mutation Burden (TMB) Calculation (`scripts/calculate_tmb.py`):**
    *   Counts SNVs per sample from MAF files.
    *   Optionally normalizes by exome size to report TMB in mutations/Megabase.

9.  **GSEA Gene Ranking (`scripts/generate_regression_ranked_list.py`):**
    *   Implements a regression-based approach (Strategy B).
    *   For each gene, performs logistic regression: `gene_mutation_status ~ signature_exposure + TMB`.
    *   Ranks genes based on the significance (signed -log10(p-value)) of the signature exposure coefficient.
    *   Outputs a `.rnk` file for GSEA.

10. **GSEA Execution (`scripts/run_gsea_analysis.py`):**
    *   Uses `gseapy` to perform GSEA Preranked analysis on the ranked gene list against a specified GMT file (e.g., MSigDB Hallmark gene sets).
    *   Generates standard GSEApy output reports and plots.

11. **GSEA Summary Visualization (`scripts/plot_gsea_summary.py`):**
    *   Creates a summary bar plot of the top N GSEA pathways by Normalized Enrichment Score (NES), with FDR q-value annotations.

12. **Automated Report Generation (`scripts/generate_report.py`):**
    *   Uses `WeasyPrint` to compile figures and placeholder text from all analysis stages into a structured PDF report.

13. **Pipeline Orchestration (`main_pipeline.py`):**
    *   A master script to run the entire pipeline or selected stages with centralized parameter management.
    *   Uses `argparse` for inputs and `subprocess.run()` to call individual scripts.
    *   Manages file paths systematically using an internal `PipelinePaths` class.

## Setup and Usage

**1. Prerequisites:**
    *   Python 3.8+
    *   Git
    *   **For PDF Report Generation (`scripts/generate_report.py`):**
        *   WeasyPrint and its system dependencies (Pango, Cairo, gdk-pixbuf). On Windows, this typically requires installing an MSYS2 environment and then installing `mingw-w64-x86_64-pango` via `pacman`, and ensuring `C:\msys64\mingw64\bin` is in the system PATH. Refer to WeasyPrint documentation for detailed OS-specific instructions.

**2. Clone the Repository:**
   ```bash
   git clone <your_repository_url>
   cd pan_cancer_signatures_jules

3. Create and Activate Virtual Environment:

python -m venv venv
# On Windows
venv\Scripts\activate.bat
# On macOS/Linux
source venv/bin/activate

4. Install Dependencies:
pip install -r requirements.txt

5. Prepare Input Data:
* Reference Genome: Download a human reference genome FASTA file (e.g., hg38.fa from UCSC or Ensembl) and note its absolute path. This project used hg38.fa.
* COSMIC Signatures: Download the COSMIC SBS signatures file (e.g., COSMIC_v3.4_SBS_GRCh38.txt) from the Sanger Institute website and place it in data/reference_data/.
* Gene Set GMT File (for GSEA): Download desired GMT files (e.g., h.all.vX.X.symbols.gmt for Hallmark gene sets) from MSigDB (Broad Institute) and place them in a directory like data/reference_data/msigdb/.

6. Running the Pipeline:
The pipeline can be run end-to-end or stage-by-stage using main_pipeline.py.
Example: Running the full pipeline for a new analysis:
python main_pipeline.py ^
  --cohort_list "COHORT1,COHORT2" ^
  --k_lda K_VALUE ^
  --lda_seed YOUR_SEED ^
  --ref_genome_fasta "PATH/TO/YOUR/hg38.fa" ^
  --cosmic_signatures_file "./data/reference_data/COSMIC_v3.4_SBS_GRCh38.txt" ^
  --gmt_file "./data/reference_data/msigdb/h.all.v2024.1.Hs.symbols.gmt" ^
  --exome_size_mb 30.0 ^
  --gsea_pairs "COHORT1:SignatureX,COHORT2:SignatureY" ^
  --base_data_dir "./data_new_run_name" ^
  --base_results_dir "./results_new_run_name" ^
  --report_output_file "./reports_new_run_name/Pipeline_Report.pdf" ^
  --run_stages "all" ^
  --regression_min_mutations 3 ^
  --gsea_plot_top_n 20 ^
  --gsea_plot_fdr_threshold 0.25 
  # Add other GSEA/diffmut parameters as needed by main_pipeline.py

(Note: ^ is for line continuation in Windows CMD. Use \ for macOS/Linux bash.)
Refer to python main_pipeline.py --help for a full list of arguments and their descriptions. Individual scripts in the scripts/ directory can also be run standalone if needed (refer to their respective --help output).

Key Findings (Summary for k=5 and k=7 Initial Runs)
k=5 Analysis (LUAD, SKCM, BRCA): Successfully identified UV (dominant in SKCM), Smoking (dominant in LUAD), APOBEC (notable in BRCA), and a mixed MMRd/Aging signature. GSEA with regression-based ranking for LUAD/Smoking highlighted Unfolded Protein Response (FDR ~0.013) and TNF-alpha Signaling via NFKB (FDR ~0.053) as negatively enriched, and EMT (FDR ~0.27) as positively enriched.
k=7 Analysis (UCEC, HNSC, STAD): Deconvolved Aging, multiple MMRd-related, POLE, APOBEC, and an HRd-like signature, showing cohort-specific distributions consistent with known tumor biology (e.g., MMRd/POLE in UCEC, APOBEC/HRd-like in HNSC). GSEA for these is ongoing/completed.
