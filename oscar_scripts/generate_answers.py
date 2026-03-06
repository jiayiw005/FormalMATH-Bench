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
from pathlib import Path
import jsonlines
import requests
from dotenv import load_dotenv
import multiprocessing as mp
import copy


load_dotenv() 


# HF Setup
hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")


def build_prompt(item):
    """Build the CoT generation prompt from an item's autoformalization field."""
    return (
        "Complete the following Lean 4 code:\n\n"
        "```lean4\n"
        + item['autoformalization']
        + "```\n\n"
        "Before producing the Lean 4 code to formally prove the given theorem, "
        "provide a detailed proof plan outlining the main proof steps and strategies.\n"
        "The plan should highlight key ideas, intermediate lemmas, and proof structures "
        "that will guide the construction of the final formal proof."
    )


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


def call_openrouter(prompt, n, temperature=1.0, top_p=0.95, model="deepseek/deepseek-prover-v2"):
    import time
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


def _save_checkpoint(checkpoint_file, lock, result_list):
    """Atomically save checkpoint to disk (process-safe via lock)."""
    with lock:
        tmp = checkpoint_file + ".tmp"
        with open(tmp, 'w') as f:
            json.dump(list(result_list), f, ensure_ascii=False, indent=2)
        os.replace(tmp, checkpoint_file)


def _dp_worker(rank, dp_size, tp_size, model_path, data_shard, num_batches,
               sampling_params_dict, processed_items, result_list,
               checkpoint_file, checkpoint_lock):
    """
    Data-parallel worker: each process owns tp_size GPUs and runs an
    independent vLLM instance with tensor_parallel_size=tp_size.
    """
    # Assign GPUs: rank 0 → GPUs [0,1], rank 1 → GPUs [2,3], etc.
    gpu_start = rank * tp_size
    gpu_ids = list(range(gpu_start, gpu_start + tp_size))
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)

    print(f"[DP worker {rank}] Using GPUs {gpu_ids}, "
          f"processing {len(data_shard)} items")

    sampling_params = SamplingParams(**sampling_params_dict)

    model = LLM(
        model=model_path,
        max_num_batched_tokens=8192,
        max_model_len=8192,
        seed=1 + rank,          # different seed per replica for diversity
        trust_remote_code=True,
        tensor_parallel_size=tp_size,
        gpu_memory_utilization=0.85,
    )

    # Get tokenizer from the vLLM instance to apply chat template
    tokenizer = model.get_tokenizer()

    for i, item in enumerate(tqdm(data_shard, desc=f"[DP-{rank}]", position=rank)):
        key = (item.get('source', ''), item.get('refined_statement', ''))
        if key in processed_items:
            continue

        raw_prompt = build_prompt(item)

        # Wrap in ChatML format so the model sees the assistant trigger
        messages = [{"role": "user", "content": raw_prompt}]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        all_answers = []

        try:
            for _ in range(num_batches):
                outputs = model.generate([prompt], sampling_params, use_tqdm=False)
                all_answers.extend([o.text for o in outputs[0].outputs])

            item['answers'] = all_answers
        except Exception as e:
            print(f"[DP worker {rank}] Error on item {i}: {e}")
            item['answers'] = []
            item['error'] = str(e)

        result_list.append(item)

        # Save checkpoint periodically so progress survives SLURM timeouts
        # Every 10 items balances safety vs I/O overhead on large runs
        if (i + 1) % 10 == 0:
            _save_checkpoint(checkpoint_file, checkpoint_lock, result_list)

    # Final save for any remaining items
    _save_checkpoint(checkpoint_file, checkpoint_lock, result_list)
    print(f"[DP worker {rank}] Done — processed {len(data_shard)} items")


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
    openrouter_model="deepseek/deepseek-prover-v2",
    tp_size=2,
    dp_size=4,
):
    """
    Process data using vLLM to generate answers.
    Uses TP+DP hybrid parallelism: dp_size workers, each with tp_size GPUs.
    Total GPUs required = tp_size * dp_size.
    
    Args:
        model_path (str): Path to the model
        input_file (str): Path to input JSON file
        output_file (str): Path to output JSON file
        batch_size (int): Used as 'n' - number of answers per batch
        num_answers (int): Total number of answers to generate per theorem
        backend (str): 'vllm' or 'openrouter'
        tp_size (int): Tensor parallelism degree (GPUs per replica)
        dp_size (int): Data parallelism degree (number of replicas)
    
    Returns:
        list: The processed data
    """
    # Setup checkpoint — derive from output_file so each run has its own
    current_directory = os.getcwd()
    output_stem = Path(output_file).stem  # e.g. "gen_goedel-v2-grind-qlora_..."
    checkpoint_file = os.path.join(current_directory, f'checkpoint_{output_stem}.json')
    
    # Read data (supports both .json arrays and .jsonl)
    print(f"Reading data from {input_file}...")
    if input_file.endswith('.jsonl'):
        data = []
        with jsonlines.open(input_file) as reader:
            for obj in reader:
                data.append(obj)
    else:
        with open(input_file, 'r') as f:
            data = json.load(f)
    
    # Calculate num_batches
    n = batch_size 
    num_batches = math.ceil(num_answers / n)
    
    # Sampling parameters as a dict (picklable for multiprocessing)
    sampling_params_dict = dict(
        temperature=0.7,          # lowered from 1.0 - formal proofs need precision
        max_tokens=8192,
        top_p=0.95,
        n=n,
        stop=["<|im_end|>", "<|endoftext|>"],  # stop at ChatML end tokens
    )
    
    # Load existing checkpoint results
    existing_results = load_checkpoint(checkpoint_file)
    processed_items = get_processed_items(existing_results)
    print(f"Loaded {len(existing_results)} previously processed items")
    
    results = list(existing_results)
    
    if backend == "vllm":
        # Validate GPU count
        num_gpus = torch.cuda.device_count()
        required_gpus = tp_size * dp_size
        if num_gpus < required_gpus:
            raise RuntimeError(
                f"Need {required_gpus} GPUs (TP={tp_size} × DP={dp_size}) "
                f"but only {num_gpus} available."
            )
        print(f"TP+DP hybrid: {dp_size} data-parallel replicas × "
              f"{tp_size}-way tensor parallelism = {required_gpus} GPUs")

        # Filter out already-processed items
        pending_data = []
        for item in data:
            key = (item.get('source', ''), item.get('refined_statement', ''))
            if key not in processed_items:
                pending_data.append(copy.deepcopy(item))

        if not pending_data:
            print("All items already processed, nothing to do.")
        else:
            # Shard data across DP workers (round-robin for balance)
            shards = [[] for _ in range(dp_size)]
            for idx, item in enumerate(pending_data):
                shards[idx % dp_size].append(item)

            for r in range(dp_size):
                print(f"  Shard {r}: {len(shards[r])} items")

            # Shared result list and lock via Manager
            manager = mp.Manager()
            result_list = manager.list()
            # Seed with existing results so checkpoint file is always complete
            result_list.extend(existing_results)
            checkpoint_lock = manager.Lock()

            # Spawn workers
            processes = []
            for rank in range(dp_size):
                p = mp.Process(
                    target=_dp_worker,
                    args=(rank, dp_size, tp_size, model_path, shards[rank],
                          num_batches, sampling_params_dict, processed_items,
                          result_list, checkpoint_file, checkpoint_lock),
                )
                p.start()
                processes.append(p)

            for p in processes:
                p.join()

            # Gather results from shared list (includes existing + new)
            results = list(result_list)

            # Final checkpoint save
            _save_checkpoint(checkpoint_file, checkpoint_lock, result_list)
            print(f"Final checkpoint saved ({len(results)} total)")
        
    elif backend == "openrouter":
        print("Running in OpenRouter mode (CPU only).")
        
        sampling_params = SamplingParams(**sampling_params_dict)
        unsaved = []
        for i in tqdm(range(len(data)), desc="Generating proofs"):
            item = data[i]
            key = (item.get('source', ''), item.get('refined_statement', ''))
            if key in processed_items:
                continue
            
            prompt = build_prompt(item)
            all_answers = []
            
            try:
                for _ in range(num_batches):
                    batch_answers = call_openrouter(
                        prompt, n=n,
                        temperature=sampling_params.temperature,
                        top_p=sampling_params.top_p,
                        model=openrouter_model
                    )
                    all_answers.extend(batch_answers)
                    
                item['answers'] = all_answers
            except Exception as e:
                print(f"Error processing item {i}: {e}")
                item['answers'] = []
                item['error'] = str(e)
            
            results.append(item)
            unsaved.append(item)
            
            if len(unsaved) % 10 == 0:
                with open(checkpoint_file, 'w') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                unsaved = []
    
    else:
        raise ValueError(f"Unknown backend: {backend}")
    
    # Save final results
    print("Saving final results...")
    with open(output_file, 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"Processing complete! Total of {len(results)} items processed")
    print(f"Final results saved to: {output_file}")
    
    return results


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
    parser.add_argument('--tp', type=int, default=2,
                        help='Tensor parallelism degree (GPUs per replica)')
    parser.add_argument('--dp', type=int, default=4,
                        help='Data parallelism degree (number of replicas)')
    
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
        tp_size=args.tp,
        dp_size=args.dp,
    )

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()