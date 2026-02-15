import re
import json
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import random
from tqdm import tqdm
import os
import argparse
import math
from multiprocessing import Pool, cpu_count, Manager
from pathlib import Path
import jsonlines
import requests
from dotenv import load_dotenv
from pathlib import Path
import multiprocessing
multiprocessing.set_start_method("spawn", force=True)


load_dotenv() 


# HF Setup
hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")

## checking for gpu
# def print_gpu_usage():
#     try:
#         gpu_id = torch.cuda.current_device()
#         props = torch.cuda.get_device_properties(gpu_id)

#         allocated = torch.cuda.memory_allocated(gpu_id) / 1024**2
#         reserved  = torch.cuda.memory_reserved(gpu_id) / 1024**2
#         total     = props.total_memory / 1024**2
#         pct = (allocated / total) * 100

#         print(f"[GPU {gpu_id}] Allocated: {allocated:.1f} MB | "
#               f"Reserved: {reserved:.1f} MB | Total: {total:.1f} MB | "
#               f"Usage: {pct:.1f}%")

#     except Exception as e:
#         print(f"GPU usage unavailable: {e}")



def init_worker(model_path, gpu_id):
    """Initialize worker process, set GPU environment, and initialize the model"""
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"Process {os.getpid()} using GPU {gpu_id}")
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.version.cuda)


    global model
    
    # prev initialization
    # model = LLM(
    #     model=model_path,
    #     max_num_batched_tokens=8192,
    #     max_model_len=8192,
    #     seed=1,
    #     trust_remote_code=True,
    #     tensor_parallel_size=1
    # )
    
    model = LLM(
        model=model_path,
        max_num_batched_tokens=2048,
        max_model_len=1024,
        seed=1,
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.85,
        enable_chunked_prefill=False,
    )

def process_single_item(item, sampling_params, num_batches, backend="vllm", openrouter_model=None, n=None):
    global model
    
    print("BACKEND =", backend)
    print("RUNNING FILE =", __file__)   
    
    item['autoformalization'] = "\nComplete the following Lean 4 code:\n```lean4\n"+item['autoformalization']
    prompt = item['autoformalization']
    
    all_answers = []
    try:
        # Original processing for vLLM and OpenRouter
        for _ in tqdm(range(num_batches), desc=f"Processing item {item.get('source','unknown')}", leave=False):
            if backend == "vllm":
                model_outputs = model.generate(
                    [prompt],
                    sampling_params,
                    use_tqdm=False
                )
                batch_answers = [output.text for output in model_outputs[0].outputs]

            elif backend == "openrouter":
                num_processes = 1
                print("OpenRouter backend: forcing num_processes=1 for safety.")
                batch_answers = call_openrouter(
                    prompt,
                    n=n,
                    temperature=sampling_params.temperature,
                    top_p=sampling_params.top_p,
                    model=openrouter_model
                )
                
            else:
                raise ValueError(f"Unknown backend: {backend}")

            all_answers.extend(batch_answers)

        item['answers'] = all_answers
        return item

    except Exception as e:
        print(f"Error in process_single_item: {e}")
        item['answers'] = []
        item['error'] = str(e)
        return item
    

def load_checkpoint(checkpoint_file):
    """Load checkpoint file"""
    try:
        with open(checkpoint_file, 'r') as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

def get_processed_items(results):
    """Get set of identifiers for processed items"""
    return {(item.get('source', ''), item.get('refined_statement', '')) for item in results}

def process_batch(args):
    (start_idx, end_idx, data, sampling_params,
     process_id, num_batches, checkpoint_dir,
     backend, openrouter_model, n) = args
    # Create a unique checkpoint file for each process
    checkpoint_file = os.path.join(checkpoint_dir, f'checkpoint_process_{process_id}.json')
    batch_results = []

    # Load this process's checkpoint
    existing_results = load_checkpoint(checkpoint_file)
    processed_items = get_processed_items(existing_results)

    for i in tqdm(range(start_idx, end_idx), desc=f"Process {os.getpid()} progress"):
        item = data[i]
        # Check if already processed
        if (item.get('source', ''), item.get('refined_statement', '')) in processed_items:
            continue
            
        result = process_single_item(
            item,
            sampling_params,
            num_batches,
            backend=backend,
            openrouter_model=openrouter_model,
            n=n
        )
        if result:
            batch_results.append(result)
            
            # Periodically save checkpoint
            if len(batch_results) % 10 == 0:  # Save every 10 items
                existing_results.extend(batch_results)
                with open(checkpoint_file, 'w') as f:
                    json.dump(existing_results, f, ensure_ascii=False, indent=2)
                batch_results = []  # Clear saved results

    # Save remaining results
    if batch_results:
        existing_results.extend(batch_results)
        with open(checkpoint_file, 'w') as f:
            json.dump(existing_results, f, ensure_ascii=False, indent=2)

    return checkpoint_file

def merge_checkpoints(checkpoint_files, output_file):
    """Merge results from all checkpoint files"""
    all_results = []
    for checkpoint_file in checkpoint_files:
        if os.path.exists(checkpoint_file):
            results = load_checkpoint(checkpoint_file)
            all_results.extend(results)
            # Optionally delete temporary checkpoint files
            # os.remove(checkpoint_file)

    # Save merged results
    with open(output_file, 'w') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    return all_results

def process_data(
    model_path,
    input_file,
    output_file,
    api_port=8012,
    num_processes=96,
    batch_size=200,
    save_interval=16,
    resume=True, 
    mode=None, 
    num_answers=3200, 
    backend="vllm",
    openrouter_model="deepseek/deepseek-prover-v2"
):
    """
    Process data using vLLM to generate answers.
    This function provides compatibility with the original pipeline interface.
    
    Args:
        model_path (str): Path to the model
        input_file (str): Path to input JSON file
        output_file (str): Path to output JSON file
        api_port (int): Not used with vLLM, kept for compatibility
        num_processes (int): Not used with vLLM, kept for compatibility
        batch_size (int): Used as 'n' - number of answers per batch
        save_interval (int): Not used with vLLM, kept for compatibility
        resume (bool): Will use checkpoint mechanism
        mode (str): Not used with vLLM, kept for compatibility
        num_answers (int): Total number of answers to generate per theorem
        
    Returns:
        list: The processed data
    """
    # Setup checkpoint directory
    current_directory = os.getcwd()
    checkpoint_dir = os.path.join(current_directory, 'checkpoint_mp')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Read data
    print(f"Reading data from {input_file}...")
    data = []
    with jsonlines.open(input_file) as reader:
        for obj in reader:
            data.append(obj)
    
    # Calculate num_batches
    n = batch_size 
    nums_answer = num_answers
    num_batches = math.ceil(nums_answer / n)
    
    # Set sampling parameters
    sampling_params = SamplingParams(
        temperature=1.0,
        max_tokens=2048,
        top_p=0.95,
        n=n,
    )
    
    if backend == "vllm":
        # detect GPUs
        available_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")

        if not available_gpus[0]:
            import torch
            available_gpus = list(range(torch.cuda.device_count()))
        else:
            available_gpus = [int(gpu) for gpu in available_gpus]

        num_gpus = len(available_gpus)
        if num_gpus == 0:
            raise RuntimeError("No available GPUs for vLLM mode.")

        print(f"Using {num_gpus} GPUs: {available_gpus}")

        # slice data per GPU
        batch_size_per_gpu = len(data) // num_gpus
        if batch_size_per_gpu == 0:
            batch_size_per_gpu = 1
            num_gpus = len(data)

        pool_args = []
        for i in range(num_gpus):
            start_idx = i * batch_size_per_gpu
            end_idx = start_idx + batch_size_per_gpu if i < num_gpus - 1 else len(data)

            pool_args.append((
                start_idx, end_idx, data,
                sampling_params, i,
                num_batches, checkpoint_dir,
                backend, openrouter_model, n
            ))

        pools = []
        tasks = []

        # One process per GPU
        for gpu_id in available_gpus[:num_gpus]:
            pool = Pool(
                processes=1,
                initializer=init_worker,
                initargs=(model_path, gpu_id)
            )
            pools.append(pool)

            task = pool.apply_async(process_batch, args=[pool_args[len(tasks)]])
            tasks.append(task)

    else:
        num_processes = 1
        print("Running in OpenRouter mode (CPU only).")

        # no GPUs, just split data among num_processes workers
        batch_size_per_proc = len(data) // num_processes
        if batch_size_per_proc == 0:
            batch_size_per_proc = 1
            num_processes = len(data)

        pool_args = []
        for i in range(num_processes):
            start_idx = i * batch_size_per_proc
            end_idx = start_idx + batch_size_per_proc if i < num_processes - 1 else len(data)

            pool_args.append((
                start_idx, end_idx, data,
                sampling_params, i,
                num_batches, checkpoint_dir,
                backend, openrouter_model, n
            ))

        pool = Pool(processes=num_processes)
        pools = [pool]
        tasks = [pool.apply_async(process_batch, args=[args]) for args in pool_args]


    # Wait for all tasks to complete and collect checkpoint file paths
    checkpoint_files = []
    for task in tqdm(tasks, desc="Waiting for tasks to complete"):
        checkpoint_file = task.get()
        checkpoint_files.append(checkpoint_file)
    
    # Close process pools
    for pool in pools:
        pool.close()
        pool.join()
    
    # Merge results from all checkpoint files
    print("Merging results...")
    final_results = merge_checkpoints(checkpoint_files, output_file)
    
    print(f"Processing complete! Total of {len(final_results)} items processed")
    print(f"Final results saved to: {output_file}")
    
    return final_results


def call_openrouter(prompt, n, temperature=1.0, top_p=0.95, model="deepseek/deepseek-prover-v2"):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ.get('OPENROUTER_API_KEY')}",
    }  
    
    print("Using OpenRouter model:", model)
    
    if not os.environ.get('OPENROUTER_API_KEY'):
        raise ValueError("OpenRouter API key not found. Please set the OPENROUTER_API_KEY environment variable.")

    payload = {
        "model": model,
        "n": n,
        "temperature": temperature,
        "top_p": top_p,
        "messages": [{"role": "user", "content": prompt}],
    }

    r = requests.post(url, headers=headers, json=payload)
    r.raise_for_status()

    data = r.json()
    time.sleep(0.4) 

    # Extract completions
    return [c["message"]["content"] for c in data["choices"]]


def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate answers using vLLM')
    
    # Define backend models
    parser.add_argument('--backend', type=str, default='vllm',
                    choices=['vllm', 'openrouter'],
                    help='Backend for generation: vllm, openrouter')
    
    parser.add_argument('--openrouter_model', type=str,
                    default='deepseek/deepseek-prover-v2',
                    help='Model name for OpenRouter API')

    parser.add_argument('--model', type=str, default=None,
                        help='Path to the model')
    parser.add_argument('--input_file', type=str, default=None,
                        help='Path to the input data file')
    parser.add_argument('--generated_file', type=str, default=None,
                        help='Path to the final output file')
    parser.add_argument('--n', type=int, default=200,
                        help='Number of answers generated per sample')
    parser.add_argument('--nums_answer', type=int, default=3200,
                        help='Total number of answers to generate per input')
    
    return parser.parse_args()



def main():
    args = parse_arguments()
    
    
    
    return process_data(
        model_path=args.model,
        input_file=args.input_file,
        output_file=args.generated_file,
        batch_size=args.n,
        num_answers=args.nums_answer,
        backend=args.backend,
        openrouter_model=args.openrouter_model,
    )

if __name__ == "__main__":
    main()