import os
import re
import matplotlib.pyplot as plt
import numpy as np

# Define the log files to parse
log_dir = "benchmark_logs"
files = {
    "Leaching (Explicit)": "leach_explicit.log",
    "Leaching (Implicit)": "leach_implicit.log",
    "Curing (Explicit)": "cure_explicit.log",
    "Curing (Implicit)": "cure_implicit.log"
}

# Data storage
total_steps = []
total_times = []
labels = list(files.keys())

# Regular Expressions to hunt for the exact data patterns in your C++ logs
# Matches: "[Saturation] Max div(q): 0.445 | Sub-steps: 160235"
regex_explicit_steps = re.compile(r"Sub-steps:\s+(\d+)")
# Matches: "[Implicit Saturation] Converged in 7 iterations"
regex_implicit_steps = re.compile(r"\[Implicit Saturation\] Converged in\s+(\d+)")
# Matches bash time output: "real    0m45.230s" or "real 45.23"
regex_time_min_sec = re.compile(r"real\s+(\d+)m([\d.]+)s")
regex_time_sec = re.compile(r"real\s+([\d.]+)")

print("--- Extracting Benchmark Data ---")

for name, filename in files.items():
    filepath = os.path.join(log_dir, filename)
    
    steps_count = 0
    time_seconds = 0.0
    
    if not os.path.exists(filepath):
        print(f"WARNING: File not found -> {filepath}")
        total_steps.append(0)
        total_times.append(0.0)
        continue

    with open(filepath, 'r') as f:
        for line in f:
            # 1. Parse Computational Steps
            if "Explicit" in name:
                match_steps = regex_explicit_steps.search(line)
                if match_steps:
                    steps_count += int(match_steps.group(1))
            else:
                match_steps = regex_implicit_steps.search(line)
                if match_steps:
                    steps_count += int(match_steps.group(1))
                    
            # 2. Parse Wall-Clock Time
            match_time_ms = regex_time_min_sec.search(line)
            if match_time_ms:
                time_seconds = (float(match_time_ms.group(1)) * 60) + float(match_time_ms.group(2))
            else:
                match_time_s = regex_time_sec.search(line)
                if match_time_s:
                    time_seconds = float(match_time_s.group(1))

    total_steps.append(steps_count)
    total_times.append(time_seconds)
    
    print(f"{name: <20} | Total Iterations/Sub-steps: {steps_count: <10} | Time: {time_seconds:.2f} s")

# ==============================================================================
# Plotting the Data
# ==============================================================================
x = np.arange(len(labels))
width = 0.35

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Total Computational Steps (Using Log Scale!)
# Because explicit steps can be 160,000+ and implicit can be ~30, a log scale is required
bars1 = ax1.bar(x, total_steps, color=['#1f77b4', '#ff7f0e', '#1f77b4', '#ff7f0e'])
ax1.set_yscale('log')
ax1.set_ylabel('Total Solver Iterations (Log Scale)', fontsize=12)
ax1.set_title('Computational Effort per Scenario', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(labels, rotation=15, ha="right")
ax1.grid(axis='y', linestyle='--', alpha=0.7)

# Add values on top of bars
for bar in bars1:
    yval = bar.get_height()
    if yval > 0:
        ax1.text(bar.get_x() + bar.get_width()/2, yval, f'{int(yval):,}', 
                 ha='center', va='bottom', fontsize=10)

# Plot 2: Wall-Clock Time
bars2 = ax2.bar(x, total_times, color=['#2ca02c', '#d62728', '#2ca02c', '#d62728'])
ax2.set_ylabel('Execution Time (seconds)', fontsize=12)
ax2.set_title('Wall-Clock Execution Time', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(labels, rotation=15, ha="right")
ax2.grid(axis='y', linestyle='--', alpha=0.7)

for bar in bars2:
    yval = bar.get_height()
    if yval > 0:
        ax2.text(bar.get_x() + bar.get_width()/2, yval + (max(total_times)*0.02), f'{yval:.1f}s', 
                 ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig("benchmark_results.png", dpi=300)
print("\nSuccess! Saved benchmark plots to 'benchmark_results.png'.")