import os
import pandas as pd
import ast
import matplotlib.pyplot as plt
import sys
from glob import glob

category = sys.argv[1]
clusters_tosketch = sys.argv[2] 
clusters_tosketch = clusters_tosketch.split(",")

n_samples = 5

def get_merged_df(csv_path):
    csv_files = glob(csv_path)
    filtered_dfs = []

    for i,file in enumerate(csv_files):
        print(f'{i*100/len(csv_files):.1f}%', file, end='\r', flush=True)
        # Read the CSV file
        df = pd.read_csv(file, on_bad_lines='skip', engine='python')

        #df = df[df['countrycode'].isin(countries)]

        # Add the filtered dataframe to the list
        filtered_dfs.append(df[['key_id', 'countrycode', 'locale', 'duration', 'drawing']])

    # Concatenate all filtered dataframes
    merged_df_info = pd.concat(filtered_dfs, ignore_index=True)
    return merged_df_info

def plot_sketches(df):
    for i, row in df.iterrows():
        key_id = row["key_id"]
        drawing = ast.literal_eval(row["drawing"])
        cluster = row["cluster"]
        plt.figure(figsize=(5, 5))
        ax = plt.gca()
        ax.set_frame_on(False)  # Remove all spines
        ax.set_xticks([])  # Remove x ticks
        ax.set_yticks([])  # Remove y ticks
        
        for e in drawing:
            plt.plot(e[0], [-_ for _ in e[1]], color="black", linewidth=4)  # Set all lines to black

        plt.tight_layout()
        plt.savefig(results_dir+f"example_{cluster}_{key_id}.png", transparent=True)
        plt.close()


results_dir = "../plots_downstream/"+category+"/examples_sketches_clusters/"

if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Iterate over csv files 
directory = f"../data/csv_files/{category}/{category}/"

df = get_merged_df(directory+'*.csv')

# Load cluster files
try:
    cluster_df = pd.read_csv(f"../data/umap_files/{category}_umap_clusters.csv")
except FileNotFoundError:
    cluster_df = pd.read_csv(f"../data/umap_files/non-clusterable/{category}_umap_clusters.csv")

cluster_df["cluster"] = cluster_df["cluster"].astype(str)

# sample n_samples from each country
for cluster in clusters_tosketch:
    cluster_df_sub = cluster_df[cluster_df['cluster'] == cluster]
    n_samples = min(n_samples, len(cluster_df_sub))
    sampled_df = cluster_df_sub.sample(n=n_samples, random_state=42)
    sampled_og_df = df[df["key_id"].isin(sampled_df["key_id"])]
    sampled_og_df["cluster"] = cluster
    plot_sketches(sampled_og_df)



