import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import seaborn as sns

# --- Load category names ---
clusterable = pd.read_csv("../data/categories_files/categories_clusterable.csv", names=["category"])
nonclusterable = pd.read_csv("../data/categories_files/categories_nonclusterable.csv", names=["category"])
categories_all = pd.concat([clusterable, nonclusterable])["category"].tolist()

# --- Load all arrays efficiently ---
arrays = []
for cat in tqdm(categories_all, desc="Loading .npy files"):
    fpath = f"./data/complexity/{cat}_duration.npy"
    if not os.path.exists(fpath):
        continue
    arr = np.load(fpath, allow_pickle=False)
    if arr.size == 0:
        continue
    arr = arr[~np.isnan(arr)]
    if arr.size > 0:
        arrays.append(arr)

# Concatenate once
if arrays:
    all_data = np.concatenate(arrays)
else:
    all_data = np.array([])

# --- Compute counts efficiently ---
if all_data.size > 0:
    total_count = all_data.size
    under20_count = np.sum((all_data >= 0) & (all_data <= 20))
    under30_count = np.sum((all_data >= 0) & (all_data <= 30))
    under0_count  = np.sum(all_data < 0)
    over30_count  = np.sum(all_data >= 30)

    pct_under20 = under20_count / total_count * 100
    pct_under30 = under30_count / total_count * 100
    pct_under0  = under0_count  / total_count * 100
    pct_over30  = over30_count  / total_count * 100

    print(f"Under 20s: {pct_under20:.10f}%", flush=True)
    print(f"Under 30s: {pct_under30:.10f}%", flush=True)
    print(f"Under 0s: {pct_under0:.10f}%", flush=True)
    print(f"Over 30s: {pct_over30:.10f}%", flush=True)

    print("max value", np.max(all_data), "min_value", np.min(all_data))
    print("mean value", np.mean(all_data), "median value", np.median(all_data))
    
    # how many and which ones are smaller than 0 and bigger than 1000s:
    print(len(all_data))
    print(len(all_data[(all_data < 0)]), flush=True)
    print(len(all_data[(all_data > 1000)]), flush=True)

    all_data_considered = all_data[(all_data >= 0) & (all_data <= 30)]
    print("mean value considered", np.mean(all_data_considered), "median value considered", np.median(all_data_considered))


    # --- Plotting ---
    plt.figure(figsize=(6,5))

    counts, edges = np.histogram(all_data[(all_data >= 0) & (all_data <= 30)], bins=50)

    plt.bar(edges[:-1], counts, width=np.diff(edges), align="edge", 
            color="lightblue", edgecolor="black")
    plt.title("")
    plt.xlabel("Duration (s)")
    plt.ylabel("Count")

    plt.tight_layout()
    sns.despine()
    plt.savefig("../plots_downstream/duration_plot.pdf")

else:
    print("No data available for plotting.")
