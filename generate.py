import re
import argparse
import logging
import asyncio
import pandas as pd
from datasets import load_dataset, Dataset
from sglang.utils import launch_server_cmd
from sglang.utils import wait_for_server, print_highlight, terminate_process
from typing_extensions import TypedDict
from typing import List, Tuple
from prompts import BASE_PROMPT_TEMPLATE, REFINE_PROMPT_TEMPLATE, SINGLE_STEP_PROMPT_TEMPLATE
import os
import openai
import subprocess
import time

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Args(TypedDict):
  base_model: str
  refine_model: str
  llm_backend: str
  hf_dataset: str
  hf_dataset_split: str
  hf_token: str
  hf_repo_id: str
  together_api_key: str
  upload_to_hf: bool
  batch_size: int
  generation_step: str
  output_dir: str
  column_to_use: str  # new parameter for defining which column to extract as problem statements
  api_key: str
  max_tokens: int

async def initialize_llm_clients(args: Args) -> Tuple[openai.AsyncOpenAI | None, openai.AsyncOpenAI | None]:
  """Initialize LLM clients based on the specified backend and generation step."""
  llm_backend = args.get("llm_backend")
  generation_step = args.get("generation_step")
  api_key = args.get("api_key")
  base_model = args.get("base_model")
  refine_model = args.get("refine_model")
  
  base_client = None
  refine_client = None
  
  if llm_backend == "together" and generation_step == "base_refine":
    base_client = openai.AsyncOpenAI(
      api_key=api_key,
      base_url="https://api.together.xyz/v1",
    )
    refine_client = openai.AsyncOpenAI(
      api_key=api_key,
      base_url="https://api.together.xyz/v1",
    )
    
  elif llm_backend == "sglang" and generation_step == "base_refine":
    hf_token = args.get("hf_token")
    if hf_token is None:
      raise ValueError("HF token is required for sglang backend")
    
    # Execute huggingface-cli login
    subprocess.run(["huggingface-cli", "login", "--token", hf_token], check=True)

    base_server_process, base_port = launch_server_cmd(
      f"python -m sglang.launch_server --model-path {base_model} --host 0.0.0.0"
    )

    wait_for_server(f"http://localhost:{base_port}")

    refine_server_process, refine_port = launch_server_cmd(
      f"python -m sglang.launch_server --model-path {refine_model} --host 0.0.0.0"
    )

    wait_for_server(f"http://localhost:{refine_port}")
    
    base_client = openai.AsyncOpenAI(
      api_key="None",
      base_url=f"http://0.0.0.0:{base_port}/v1",
    )
    
    refine_client = openai.AsyncOpenAI(
      api_key="None",
      base_url=f"http://0.0.0.0:{refine_port}/v1",
    )
    
  elif llm_backend == "sglang" and generation_step == "refine_only":
    hf_token = args.get("hf_token")
    if hf_token is None:
      raise ValueError("HF token is required for sglang backend")
    
    # Execute huggingface-cli login
    subprocess.run(["huggingface-cli", "login", "--token", hf_token], check=True)
    
    # Launch sglang server for the refine model on port 3001
    refine_server_process, refine_port = launch_server_cmd(
      f"python -m sglang.launch_server --model-path {refine_model} --host 0.0.0.0"
    )

    wait_for_server(f"http://localhost:{refine_port}")
    
    refine_client = openai.AsyncOpenAI(
      api_key="None",
      base_url=f"http://0.0.0.0:{refine_port}/v1",
    )
    
  elif llm_backend == "together" and generation_step == "refine_only":
    refine_client = openai.AsyncOpenAI(
      api_key=api_key,
      base_url="https://api.together.xyz/v1",
    )
  
  return base_client, refine_client

async def generate_content(problem_statement: str, base_client, refine_client, args: Args) -> Tuple[str, str]:
  """Generate content using the specified LLM backend."""
  llm_backend = args.get("llm_backend")
  generation_step = args.get("generation_step")
  max_tokens = args.get("max_tokens")
  if llm_backend not in ["together", "sglang"]:
    raise ValueError("LLM Backend must be either 'together' or 'sglang'")
  
  if generation_step not in ["base_refine", "refine_only"]:
    raise ValueError("Generation Step must be either 'base_refine' or 'refine_only'")
  
  base_model = args.get("base_model")
  refine_model = args.get("refine_model")
  
  try:
    BASE_PROMPT = BASE_PROMPT_TEMPLATE.format(problem_statement=problem_statement)
    
    if generation_step == "base_refine":
      # Generate the base generation
      response = await base_client.completions.create(
        model=base_model,
        prompt=BASE_PROMPT,
        max_tokens=max_tokens,
      )
      
      base_generation = response.choices[0].text
      REFINE_PROMPT = REFINE_PROMPT_TEMPLATE.format(
        original_problem_statement=problem_statement,
        original_reasoning_and_solution=base_generation
      )
      
      # Generate the refine generation
      response = await refine_client.chat.completions.create(
        model=refine_model,
        messages=[
          {"role": "system", "content": "You are a helpful assistant."},
          {"role": "user", "content": REFINE_PROMPT},
        ],
        max_tokens=max_tokens,
      )
      
      refine_generation = response.choices[0].message.content
      logging.info(f"Generated content for: {problem_statement}")
      return refine_generation, base_generation
    
    elif generation_step == "refine_only":
      REFINE_PROMPT = SINGLE_STEP_PROMPT_TEMPLATE.format(problem_statement=problem_statement)
      
      # Generate the refine generation directly
      response = await refine_client.chat.completions.create(
        model=refine_model,
        messages=[
          {"role": "system", "content": "You are a helpful assistant."},
          {"role": "user", "content": REFINE_PROMPT},
        ],
        max_tokens=max_tokens,
      )
      
      refine_generation = response.choices[0].message.content
      logging.info(f"Generated content for: {problem_statement}")
      return refine_generation, None
  
  except Exception as e:
    logging.error(f"Generation failed: {e}")
    return None, None

async def sample_extraction(refine_generation: str) -> Tuple[str | None, str | None]:
  """Extract content between <think> and </think> and <Solution> and </Solution>."""
  think_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
  solution_pattern = re.compile(r"<Solution>(.*?)</Solution>", re.DOTALL)

  try:
    think_match = think_pattern.search(refine_generation)
    solution_match = solution_pattern.search(refine_generation)

    think_content = think_match.group(1).strip() if think_match else None
    solution_content = solution_match.group(1).strip() if solution_match else None

    return think_content, solution_content
  except Exception as e:
    logging.error(f"Extraction failed: {e}")
    return None, None

async def save_to_hf(args: Args, problem_statements: List[str], reasoning_list: List[str], answer_list: List[str]):
  """Save the generated data to a Hugging Face dataset."""
  # Create dataset from dictionary
  ds = Dataset.from_dict({
    "problem_statement": problem_statements,
    "reasoning": reasoning_list,
    "answer": answer_list
  })
  
  # upload to huggingface
  hf_token = args.get("hf_token")
  hf_repo_id = args.get("hf_repo_id")
  if not all([hf_token, hf_repo_id]):
    raise ValueError("HF Token and HF Repo ID are required")
  ds.push_to_hub(hf_repo_id, token=hf_token)

async def main(args: Args):
  """Main function to orchestrate data generation and saving."""
  # Initialize LLM clients only once
  base_client, refine_client = await initialize_llm_clients(args)

  # Fetch problem statements from a Hugging Face dataset if column_to_use is provided
  column_to_use = args.get("column_to_use")
  if column_to_use:
    hf_dataset = args.get("hf_dataset")
    hf_dataset_split = args.get("hf_dataset_split")
    logging.info(f"Loading dataset {hf_dataset} (split: {hf_dataset_split})...")
    ds = load_dataset(hf_dataset, split=hf_dataset_split)
    if column_to_use not in ds.column_names:
      raise ValueError(f"Column '{column_to_use}' not found in dataset. Available columns: {ds.column_names}")
    problem_statements = ds[column_to_use]
  else:
    # Fallback sample if no column specified.
    problem_statements = [
      "What is the capital of France?"
    ]
  
  # Limit the number of rows if 'num_rows' is specified
  num_rows = args.get("num_rows")
  if num_rows is not None:
    problem_statements = problem_statements[:num_rows]
  
  extracted_data = []
  
  batch_size = args.get("batch_size")   # Adjust batch size as needed
  for i in range(0, len(problem_statements), batch_size):
    batch_statements = problem_statements[i:i + batch_size]
    
    # Generate content for the batch using already initialized clients
    generation_tasks = [
      generate_content(statement, base_client, refine_client, args)
      for statement in batch_statements
    ]
    results = await asyncio.gather(*generation_tasks)
    
    for problem_statement, (refine_generation, base_generation) in zip(batch_statements, results):
      if refine_generation is None:
        logging.warning(f"Skipping extraction for problem: {problem_statement} due to generation failure.")
        extracted_data.append({"problem_statement": problem_statement, "reasoning": None, "answer": None})
        continue
      
      # Extract samples
      think_content, solution_content = await sample_extraction(refine_generation)
      
      if think_content and solution_content:
        extracted_data.append({
          "problem_statement": problem_statement,
          "reasoning": think_content,
          "answer": solution_content
        })
      else:
        logging.warning(f"Could not extract think or solution for problem: {problem_statement}")
        extracted_data.append({"problem_statement": problem_statement, "reasoning": None, "answer": None})

  df = pd.DataFrame(extracted_data)
  output_dir = args.get("output_dir")
  try:
    os.makedirs(output_dir, exist_ok=True)
    df.to_excel(os.path.join(output_dir, "synthetic_dataset.xlsx"), index=False)
    logging.info("Data saved to synthetic_dataset.xlsx")
  except Exception as e:
    logging.error(f"Failed to save data to Excel: {e}")

  if args.get("upload_to_hf"):
    try:
      reasoning_list = [row["reasoning"] for row in extracted_data]
      answer_list = [row["answer"] for row in extracted_data]
      await save_to_hf(args, problem_statements, reasoning_list, answer_list)
      logging.info("Data uploaded to Hugging Face Hub.")
    except Exception as e:
      logging.error(f"Failed to upload to Hugging Face: {e}")

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Generate synthetic dataset and optionally upload to Hugging Face.")
  parser.add_argument("--base_model", type=str, help="Base model name")
  parser.add_argument("--refine_model", type=str, required=True, help="Refine model name")
  parser.add_argument("--llm_backend", type=str, choices=["together", "sglang"], required=True, help="LLM backend to use (together or sglang)")
  parser.add_argument("--hf_dataset", type=str, default="AI-MO/NuminaMath-TIR", help="Hugging Face dataset name")
  parser.add_argument("--hf_dataset_split", type=str, default="train", help="Hugging Face dataset split")
  parser.add_argument("--hf_token", type=str, help="Hugging Face token")
  parser.add_argument("--hf_repo_id", type=str, help="Hugging Face repo ID")
  parser.add_argument("--api_key", type=str, default=None, help="Together API key")
  parser.add_argument("--upload_to_hf", action="store_true", help="Upload data to Hugging Face")
  parser.add_argument("--batch_size", type=int, default=1, help="Batch size for processing problem statements")
  parser.add_argument("--generation_step", type=str, choices=["base_refine", "refine_only"], default="base_refine", help="Choose generation step: base_refine or refine_only")
  parser.add_argument("--output_dir", type=str, default="output", help="Output directory for saving the dataset")
  parser.add_argument("--column_to_use", type=str, default=None, help="Dataset column to use as problem statement")
  parser.add_argument("--num_rows", type=int, default=10, help="Number of rows to generate")
  parser.add_argument("--max_tokens", type=int, default=1024, help="Number of tokens to generate")
  
  args = parser.parse_args()
  
  asyncio.run(main(vars(args)))
