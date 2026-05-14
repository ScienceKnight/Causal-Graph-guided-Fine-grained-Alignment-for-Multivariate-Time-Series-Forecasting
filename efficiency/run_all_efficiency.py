import subprocess
import sys

def run_script(script_name):
    try:
        subprocess.check_call([sys.executable, script_name])
        print(f" {script_name} executed successfully")
    except subprocess.CalledProcessError as e:
        print(f" {script_name} failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    scripts = [
        "setup_efficiency.py",
        "efficiency_benchmark.py",
        "plot_efficiency.py"
    ]
    
    for script in scripts:
        run_script(script)
    
    print("\n All efficiency experiments completed Results saved to:")
    print("   - efficiency_results.json")
    print("   - efficiency_report.txt")
    print("   - throughput_plot.png")
    print("   - avg_time_plot.png")