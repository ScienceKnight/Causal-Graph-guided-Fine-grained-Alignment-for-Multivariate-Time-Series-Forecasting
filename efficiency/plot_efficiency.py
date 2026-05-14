import json
import matplotlib.pyplot as plt
import numpy as np

def load_results(file_path="efficiency_results.json"):
    with open(file_path, "r") as f:
        return json.load(f)

def plot_throughput(results):
    plt.rcParams["figure.figsize"] = (12, 8)
    fig, axes = plt.subplots(2, 2, sharex=True, sharey=False)
    
    dataset_names = list(results.keys())
    model_sizes = ["base", "large"]
    sample_sizes = list(next(iter(results.values())).keys())
    batch_sizes = [res["batch_size"] for res in next(iter(next(iter(next(iter(results.values())).values())).values()))]
    
    for i, dataset in enumerate(dataset_names):
        for j, sample_size in enumerate(sample_sizes[:2]):
            ax = axes[i][j]
            for model in model_sizes:
                throughputs = [res["throughput"] for res in results[dataset][sample_size][model]]
                ax.plot(batch_sizes, throughputs, marker="o", label=f"Model: {model}")
            
            ax.set_title(f"{dataset} - Sample Size: {sample_size}")
            ax.set_xlabel("Batch Size")
            ax.set_ylabel("Throughput (samples/s)")
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("throughput_plot.png")
    plt.close()

def plot_avg_time(results):
    plt.rcParams["figure.figsize"] = (12, 8)
    fig, axes = plt.subplots(2, 2, sharex=True, sharey=False)
    
    dataset_names = list(results.keys())
    model_sizes = ["base", "large"]
    sample_sizes = list(next(iter(results.values())).keys())
    batch_sizes = [res["batch_size"] for res in next(iter(next(iter(next(iter(results.values())).values())).values()))]
    
    for i, dataset in enumerate(dataset_names):
        for j, sample_size in enumerate(sample_sizes[:2]):
            ax = axes[i][j]
            for model in model_sizes:
                avg_times = [res["avg_time_per_sample"] for res in results[dataset][sample_size][model]]
                ax.plot(batch_sizes, avg_times, marker="s", label=f"Model: {model}")
            
            ax.set_title(f"{dataset} - Sample Size: {sample_size}")
            ax.set_xlabel("Batch Size")
            ax.set_ylabel("Avg Time per Sample (s)")
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("avg_time_plot.png")
    plt.close()

if __name__ == "__main__":
    results = load_results()
    plot_throughput(results)
    plot_avg_time(results)