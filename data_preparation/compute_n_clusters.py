import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

def count_clusters(path):
    results = []
    for file in os.listdir(path):
        if "umap_clusters.csv" in file:
            df = pd.read_csv(os.path.join(path, file))
            cat = file.split("_umap_clusters.csv")[0]
            # exclude -1 (both int and str, just in case)
            clusters = [x for x in df["cluster"].unique() if x not in [-1, "-1"]]
            n_clusters = len(clusters)
            results.append([cat, n_clusters])
    return results

# Collect from both folders
n_clusters_list = []
n_clusters_list.extend(count_clusters("../data/umap_files/"))
n_clusters_list.extend(count_clusters("../data/umap_files/non-clusterable/"))

# Build dataframe
df_n_clusters = pd.DataFrame(n_clusters_list, columns=["category", "n_clusters"])

# # Summary
# print(df_n_clusters["n_clusters"].describe(), flush=True)
# print(df_n_clusters["n_clusters"].value_counts(), flush=True)
# print(df_n_clusters[df_n_clusters["n_clusters"]==df_n_clusters["n_clusters"].min()], flush=True)
# print(df_n_clusters[df_n_clusters["n_clusters"]==9], flush=True)
# print(df_n_clusters[df_n_clusters["n_clusters"]==df_n_clusters["n_clusters"].max()], flush=True)

# Plot distribution
plt.figure(figsize=(10,6))
sns.histplot(df_n_clusters["n_clusters"], discrete=True, color="darkorange", alpha=0.7)

# Ensure all unique values appear as x-ticks
unique_values = sorted(df_n_clusters["n_clusters"].unique())
plt.xticks(unique_values, fontsize=12)

plt.xlabel("Number of clusters", fontsize=16)
plt.ylabel("Frequency", fontsize=16)
plt.yticks(fontsize=12)
sns.despine()
plt.savefig("../plots_downstream/number_clusters_distribution.pdf")
plt.close()
