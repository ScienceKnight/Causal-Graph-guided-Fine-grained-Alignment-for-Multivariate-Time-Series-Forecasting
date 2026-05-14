import time
import json
import os
import numpy as np
from casualalign import CasualAligner
from datasets import load_dataset

def load_test_data(dataset_name, split="test", sample_size=1000):
    dataset = load_dataset(dataset_name, split=split)
    dataset = dataset.shuffle(seed=42).select(range(sample_size))
    src_texts = [item["source"] for item in dataset]
    tgt_texts = [item["target"] for item in dataset]
    return src_texts, tgt_texts

def measure_inference_time(aligner, src_texts, tgt_texts, batch_sizes):
    results = []
    for batch_size in batch_sizes:
        total_time = 0.0
        total_samples = 0
        for i in range(0, len(src_texts), batch_size):
            batch_src = src_texts[i:i+batch_size]
            batch_tgt = tgt_texts[i:i+batch_size]
            start_time = time.perf_counter()
            aligner.align_batch(batch_src, batch_tgt)
            end_time = time.perf_counter()
            batch_time = end_time - start_time
            total_time += batch_time
            total_samples += len(batch_src)
        avg_time_per_sample = total_time / total_samples
        throughput = total_samples / total_time
        results.append({
            "batch_size": batch_size,
            "avg_time_per_sample": avg_time_per_sample,
            "throughput": throughput,
            "total_time": total_time,
            "total_samples": total_samples
        })
    return results

def run_efficiency_experiment():
    dataset_names = ["wmt14", "flores200"]
    sample_sizes = [100, 500, 1000]
    batch_sizes = [1, 8, 16, 32, 64]
    model_sizes = ["base", "large"]
    
    all_results = {}
    
    for dataset_name in dataset_names:
        all_results[dataset_name] = {}
        for sample_size in sample_sizes:
            all_results[dataset_name][sample_size] = {}
            src_texts, tgt_texts = load_test_data(dataset_name, sample_size=sample_size)
            for model_size in model_sizes:
                aligner = CasualAligner(model_size=model_size, device="cuda" if torch.cuda.is_available() else "cpu")
                results = measure_inference_time(aligner, src_texts, tgt_texts, batch_sizes)
                all_results[dataset_name][sample_size][model_size] = results
    
    with open("efficiency_results.json", "w") as f:
        json.dump(all_results, f, indent=4)
    
    generate_summary_report(all_results)

def generate_summary_report(results):
    report = []
    report.append("=== CasualAlign Efficiency Benchmark Report ===")
    for dataset, sample_data in results.items():
        report.append(f"\nDataset: {dataset}")
        for sample_size, model_data in sample_data.items():
            report.append(f"  Sample Size: {sample_size}")
            for model_size, batch_results in model_data.items():
                report.append(f"    Model Size: {model_size}")
                for res in batch_results:
                    report.append(f"      Batch Size {res['batch_size']}:")
                    report.append(f"        Avg Time per Sample: {res['avg_time_per_sample']:.6f}s")
                    report.append(f"        Throughput: {res['throughput']:.2f} samples/s")
                    report.append(f"        Total Time: {res['total_time']:.2f}s")
    
    with open("efficiency_report.txt", "w") as f:
        f.write("\n".join(report))

if __name__ == "__main__":
    import torch
    run_efficiency_experiment()