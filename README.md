# HF2Reasoning

HF2Reasoning is a lightweight tool that converts any Hugging Face dataset into a reasoning dataset. By leveraging Large Language Models (LLMs), it generates detailed reasoning chains and solutions for problem statements. The final output is saved as an Excel file, and you can also upload the dataset to the Hugging Face Hub. This repo also incorporates the **BARE** technique for enhanced synthetic data generation.

## Installation

### Prerequisites

- Python 3.8 or later
- [pip](https://pip.pypa.io)
- [git](https://git-scm.com)

### Setup

This repo includes an installation shell script (`install.sh`) that installs all required dependencies and sets up the environment. If needed, change the script's execution permission before running it:

```bash
chmod +x install.sh
```

Then, execute the script:

```bash
./install.sh
```

The script will upgrade pip, install the necessary Python packages, clone the required sglang repository (from the v0.4.3.post2 branch), and install all components needed for HF2Reasoning.

## What It Does

HF2Reasoning processes a Hugging Face dataset by:

- **Loading a Dataset:**  
  It loads a specified Hugging Face dataset (default: `"AI-MO/NuminaMath-TIR"`) and extracts problem statements from a user-defined column.

- **Generating Reasoning and Solutions:**  
  Based on the chosen LLM backend (either `together` or `sglang`), HF2Reasoning supports two generation strategies:
  - **Base & Refine (base_refine):**  
    First generates diverse but potentially lower-quality responses using a base model, then refines each response with an instruction-tuned model.
  - **Refine Only (refine_only):**  
    Directly generates refined responses.
  
  The output contains a reasoning chain (text between `<think>` and `</think>`) and a solution (text between `<Solution>` and `</Solution>`).

- **Saving the Output:**  
  The generated dataset is saved as an Excel file (`synthetic_dataset.xlsx`) in your chosen output directory.

- **Optional Hugging Face Upload:**  
  If provided with a valid Hugging Face token and repository ID, the dataset can be uploaded directly to the Hugging Face Hub.

## BARE: Combining Base and Instruction-Tuned Language Models

HF2Reasoning employs the **BARE** technique—**Base-Refine**—to achieve the best of both worlds:

- **Instruct Models:** Produce higher quality but less diverse responses.
- **Base Models:** Offer more diverse, though sometimes lower quality, outputs.
- **BARE:** Combines these strengths by generating diverse synthetic data with a base model and refining it with an instruction-tuned model.

For more details on this concept, see:

- **Paper:** [BARE: Combining Base and Instruction-Tuned Language Models for Better Synthetic Data Generation](https://arxiv.org/abs/2502.01697)
- **Code:** [BARE: Github](https://github.com/pgasawa/BARE)
- **Thread:** [X Thread Post](https://x.com/pgasawa/status/1887201938607120592)

This approach not only boosts data quality but also enhances diversity, which has proven beneficial for downstream tasks such as fine-tuning models for reasoning and evaluation.

## **Support for sglang**  

HF2Reasoning fully supports `sglang`, an efficient and scalable framework for serving large language models. When using `sglang` as the LLM backend, the tool automatically launches local inference servers for both the base and refine models, ensuring seamless and high-speed generation of reasoning datasets. This integration allows users to leverage local models for privacy-sensitive or cost-effective workflows, avoiding reliance on external APIs. Additionally, `sglang` makes it easier to fine-tune and experiment with different models, providing a flexible and adaptable environment for synthetic data generation.

## Execution Example

Once installed, you can run HF2Reasoning with the following command:

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

### Command-line Arguments Overview

- `--base_model`: *(Optional)* Base model for the base_refine strategy.
- `--refine_model` (required): Model used for refining the generated output.
- `--llm_backend` (required): LLM backend to use (choose between `together` or `sglang`).
- `--hf_dataset`: Hugging Face dataset name (default: `"AI-MO/NuminaMath-TIR"`).
- `--hf_dataset_split`: Dataset split (default: `"train"`).
- `--hf_token`: Hugging Face token (required for certain backends).
- `--hf_repo_id`: Repository ID for dataset uploads.
- `--api_key`: API key for the Together backend (if applicable).
- `--upload_to_hf`: Flag to upload the generated dataset to Hugging Face.
- `--batch_size`: Number of problem statements processed per batch.
- `--generation_step`: Generation strategy (`base_refine` or `refine_only`).
- `--output_dir`: Directory where the Excel file is saved.
- `--column_to_use`: Dataset column containing problem statements.
- `--num_rows`: Number of rows to process.
- `--max_tokens`: Maximum tokens to generate per prompt.

## License

> Note: The BARE component included in the `bare/` directory is licensed under Apache 2.0.

## Owner

HF2Reasoning is maintained and owned by Syed Hasan Abbas.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request if you have suggestions, improvements, or bug fixes.
