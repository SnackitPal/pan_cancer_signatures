import requests
import argparse
import os
import json

def download_file(url, filepath):
    """Downloads a file from a URL and saves it to filepath in chunks."""
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(filepath, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"Successfully downloaded: {filepath}")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")
    except IOError as e:
        print(f"Error saving file {filepath}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Download TCGA MAF files from GDC API.")
    parser.add_argument("--project_ids", required=True, type=str,
                        help="Comma-separated string of TCGA project IDs (e.g., 'TCGA-LUAD,TCGA-SKCM').")
    parser.add_argument("--output_dir", required=True, type=str,
                        help="Base output directory for MAF files (e.g., './data/raw_mafs/').")

    args = parser.parse_args()
    project_ids_list = [pid.strip() for pid in args.project_ids.split(',')]
    base_output_dir = args.output_dir

    print(f"Starting MAF file download for projects: {', '.join(project_ids_list)}")
    print(f"Output directory: {base_output_dir}")

    files_endpoint = "https://api.gdc.cancer.gov/files"
    data_endpoint_base = "https://api.gdc.cancer.gov/data/"

    for project_id in project_ids_list:
        print(f"\nProcessing project: {project_id}")
        project_output_dir = os.path.join(base_output_dir, project_id)
        os.makedirs(project_output_dir, exist_ok=True)
        print(f"Output for {project_id} will be saved to: {project_output_dir}")

        file_uuids = []
        current_from = 0
        page_size = 100 # GDC API max size is often 100 or can be higher, check API docs

        while True:
            filters = {
                "op": "and",
                "content": [
                    {"op": "in", "content": {"field": "cases.project.project_id", "value": [project_id]}},
                    {"op": "in", "content": {"field": "access", "value": ["open"]}},
                    {"op": "in", "content": {"field": "data_category", "value": ["Simple Nucleotide Variation"]}},
                    {"op": "in", "content": {"field": "data_type", "value": ["Masked Somatic Mutation"]}},
                    {"op": "in", "content": {"field": "experimental_strategy", "value": ["WXS"]}}
                ]
            }
            params = {
                "filters": json.dumps(filters),
                "fields": "file_id,file_name", # Specify fields to return
                "format": "JSON",
                "size": page_size,
                "from": current_from
            }

            try:
                print(f"Fetching file list for {project_id} (from: {current_from}, size: {page_size})...")
                response = requests.get(files_endpoint, params=params)
                response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
                response_json = response.json()

                hits = response_json.get("data", {}).get("hits", [])
                if not hits:
                    if not file_uuids: # Only print "no files found" if it's the first query and nothing was found yet
                        print(f"No MAF files found for project {project_id} with current filters.")
                    break # Exit loop if no more files

                for hit in hits:
                    file_uuids.append({"id": hit["file_id"], "name": hit["file_name"]})

                # Check pagination
                total_files = response_json.get("data", {}).get("pagination", {}).get("total", 0)
                current_from += len(hits)
                if current_from >= total_files:
                    break # Exit if all files have been fetched

            except requests.exceptions.RequestException as e:
                print(f"Error fetching file list for {project_id}: {e}")
                break # Stop processing this project on error
            except json.JSONDecodeError:
                print(f"Error decoding JSON response for {project_id}.")
                break

        if file_uuids:
            print(f"Found {len(file_uuids)} MAF files for project {project_id}.")
            for file_info in file_uuids:
                file_id = file_info["id"]
                file_name = file_info["name"]
                target_file_path = os.path.join(project_output_dir, file_name)
                print(f"Downloading: {file_name} (ID: {file_id}) to {target_file_path}")
                download_url = f"{data_endpoint_base}{file_id}"
                download_file(download_url, target_file_path)
        elif not file_uuids and current_from == 0 : # Double check if no files were found from the start
             print(f"No MAF files found for project {project_id} after checking all pages or initial query.")


    print("\nAll specified projects processed.")

if __name__ == "__main__":
    main()
