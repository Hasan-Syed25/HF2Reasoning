# HF2Reasoning

HF2Reasoning is a lightweight tool that converts any Hugging Face dataset into a reasoning dataset. It leverages Large Language Models (LLMs) to generate detailed reasoning chains and solutions for problem statements in your dataset, saving the output as an Excel file and optionally uploading it to the Hugging Face Hub.

## Installation

### Prerequisites

- Python 3.8 or later
- [pip](https://pip.pypa.io)
- [git](https://git-scm.com)

### Setup

This repo includes an installation shell script (`install.sh`) that installs all the required dependencies and sets up the environment. If needed, change the script's execution permission before running it:

```bash
chmod +x install.sh
```

Then, run the script:

```bash
./install.sh
```

The script will upgrade pip, install necessary Python packages, clone the required sglang repository (from the v0.4.3.post2 branch), and install all components needed for HF2Reasoning to work.

## What It Does

HF2Reasoning processes a Hugging Face dataset by:

- **Loading a Dataset:**  
  The tool loads a specified Hugging Face dataset (default is `"AI-MO/NuminaMath-TIR"`) and extracts problem statements from a user-defined column.

- **Generating Reasoning and Solutions:**  
  Using your chosen LLM backend (either `together` or `sglang`), HF2Reasoning generates:
  - **Base and Refine (base_refine):**  
    A two-step process where a base response is generated and then refined.
  - **Refine Only (refine_only):**  
    A single-step process that directly generates a refined response.
  
  The generated text includes a reasoning chain (text between `<think>` and `</think>`) and a solution (text between `<Solution>` and `</Solution>`).

- **Saving the Output:**  
  The extracted data is saved as an Excel file (`synthetic_dataset.xlsx`) in the designated output directory.

- **Optional Hugging Face Upload:**  
  With a valid Hugging Face token and repository ID, you can choose to upload the generated dataset directly to the Hugging Face Hub.

## Execution Example

After installation, you can run the tool using the following command:

```bash
python generate.py \
  --refine_model "your_refine_model" \
  --llm_backend sglang \
  --hf_token "your_hf_token" \
  --hf_repo_id "your_hf_repo_id" \
  --column_to_use "problem_statement" \
  --num_rows 10 \
  --generation_step base_refine \
  --output_dir output \
  --max_tokens 1024
```

### Command-line Arguments:

- `--base_model`: (Optional) Name of the base model to use (only needed for the `base_refine` strategy).
- `--refine_model` (required): The model to use for refining the generated output.
- `--llm_backend` (required): Backend to use for the LLM (choose between `together` or `sglang`).
- `--hf_dataset`: Name of the Hugging Face dataset (default: `"AI-MO/NuminaMath-TIR"`).
- `--hf_dataset_split`: Dataset split to load (default: `"train"`).
- `--hf_token`: Hugging Face token (required for certain backends).
- `--hf_repo_id`: Hugging Face repository ID for dataset uploads.
- `--api_key`: API key for the Together backend (if using `together`).
- `--upload_to_hf`: Flag to upload the generated dataset to the Hugging Face Hub.
- `--batch_size`: Number of problem statements processed per batch.
- `--generation_step`: Generation strategy to use (`base_refine` or `refine_only`).
- `--output_dir`: Directory where the output Excel file will be saved.
- `--column_to_use`: Dataset column containing the problem statements.
- `--num_rows`: Number of rows to process from the dataset.
- `--max_tokens`: Maximum number of tokens to generate for each prompt.

## License

This project is open source and available under the MIT License.  
[MIT License](LICENSE)

## Owner

HF2Reasoning is maintained and owned by Syed Hasan Abbas.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request if you have suggestions or improvements.
