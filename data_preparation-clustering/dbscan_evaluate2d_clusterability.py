#!/usr/bin/env python
# coding: utf-8

import os
import pandas
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")

tag = "2d_"

data_dir = "../data/umap_files/"

list_categories = []
for file in os.listdir(data_dir):
    if file.startswith(f'2d_dbscan_dbcv_study'):
        category = file.split("_train_startseed_")[0].split("2d_dbscan_dbcv_study_")[1]
        list_categories.append(category)

list_categories = list(set(list_categories))

def get_best_dbcv(category):
    all_data = []
    for file in os.listdir(data_dir):
        if file.startswith(f'2d_dbscan_dbcv_study_{category}_train_startseed_'):
            df = pd.read_csv(data_dir+file)
            all_data.append(df)

    if len(all_data) == 0:
        print(f"No data for {category}", flush=True)
        return None
    
    df = pd.concat(all_data)
    df_grouped = df.groupby('d')['dbcv_metric'].median()
    best_dist = str(df_grouped.idxmax())
    if len(best_dist) < 4:
        best_dist = best_dist + "0"
    return best_dist

def is_surrounded_by_empty(i, j, H):
    # Get the dimensions of the histogram
    rows, cols = H.shape
    # Check the surrounding cells
    for x in range(max(0, i-1), min(rows, i+2)):
        for y in range(max(0, j-1), min(cols, j+2)):
            if H[x, y] > 0 and (x != i or y != j):
                return False
    return True


def remove_outliers(df_umap):
    ## remove outliers surrounded by empty space
    X = df_umap[['UMAP_1', 'UMAP_2']].values

    ### Define the grid size
    grid_size = 100  # Adjust this value as needed
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    x_grid = np.linspace(x_min, x_max, grid_size)
    y_grid = np.linspace(y_min, y_max, grid_size)

    ### Create a 2D histogram to count points in each grid cell
    H, xedges, yedges = np.histogram2d(X[:, 0], X[:, 1], bins=(x_grid, y_grid))

    ### Define a function to check if a cell is surrounded by empty cells

    ### Create a mask for cells surrounded by empty cells
    surrounded_by_empty_mask = np.zeros_like(H, dtype=bool)
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            if H[i, j] > 0 and is_surrounded_by_empty(i, j, H):
                surrounded_by_empty_mask[i, j] = True

    ### Find the points that are not surrounded by empty cells
    x_idx = np.digitize(X[:, 0], xedges) - 1
    y_idx = np.digitize(X[:, 1], yedges) - 1

    ### Ensure indices are within bounds
    x_idx[x_idx == H.shape[0]] = H.shape[0] - 1
    y_idx[y_idx == H.shape[1]] = H.shape[1] - 1
    dense_points_mask = ~surrounded_by_empty_mask[x_idx, y_idx]

    ### Filter the points
    df_umap = df_umap.iloc[dense_points_mask]

    return df_umap

def get_big_clusters(df_umap, best_dist, threshold=0.01):

    lower_threshold_filter = threshold*df_umap.shape[0]
    # if the number of points in a cluster is less than the lower_threshold_filter, put as noise (-1)
    df_top_clusters = df_umap.copy()
    cluster_counts = df_top_clusters[f"DBSCAN (d={best_dist})"].value_counts()
    tot_n_clusters = len(cluster_counts)
    df_top_clusters[f"DBSCAN (d={best_dist})"] = df_top_clusters[f"DBSCAN (d={best_dist})"].apply(lambda x: x if x != -1 and cluster_counts.get(x, 1) > lower_threshold_filter else -1)
    print(df_top_clusters[f"DBSCAN (d={best_dist})"].value_counts())
    return df_top_clusters, tot_n_clusters

def plot_umap(df_umap, alpha_par, category):
    # Plot UMAP and clusters
    if not os.path.exists(f"../plots_downstream/{category}"):
        os.makedirs(f"../plots_downstream/{category}")

    fig, ax = plt.subplots(figsize=(10,10))
    sns.scatterplot(x="UMAP_1", y="UMAP_2", data=df_umap, alpha=alpha_par, s=1, color="black")  
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')

    plt.savefig(f"../plots_downstream/{category}/umap_all.png")
    plt.close()


def plot_umap_clusters(df_top_clusters, alpha_par, category, best_dist):
    fig, ax = plt.subplots(figsize=(10,10))
    sns.scatterplot(x="UMAP_1", y="UMAP_2", data=df_top_clusters[df_top_clusters[f"DBSCAN (d={best_dist})"] == -1], alpha=alpha_par, s=1, color="grey")
    sns.scatterplot(x="UMAP_1", y="UMAP_2", data=df_top_clusters[df_top_clusters[f"DBSCAN (d={best_dist})"] != -1], alpha=alpha_par, s=1, hue=f"DBSCAN (d={best_dist})", palette="tab20")  
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')

    lgd = plt.legend(title="Cluster", fontsize=10, title_fontsize=10, markerscale=5)
    for lh in lgd.legendHandles: 
        lh.set_alpha(1)

    plt.savefig(f"../plots_downstream/{category}/umap_top_clusters.png")
    plt.close()

def get_perc_noise(df_top_clusters, best_dist):
    # Compute % noise
    noise = df_top_clusters[df_top_clusters[f"DBSCAN (d={best_dist})"] == -1].shape[0]
    total = df_top_clusters.shape[0]
    return noise/total

def compute_avg_variance(df_top_clusters, best_dist):
    variances = []
    for cluster in list(df_top_clusters[f"DBSCAN (d={best_dist})"].unique()):
        cluster = float(cluster)
        df_umap_cluster = df_top_clusters[df_top_clusters[f"DBSCAN (d={best_dist})"] == cluster]
        variances.append(df_umap_cluster[['UMAP_1', 'UMAP_2']].var().mean()*df_umap_cluster.shape[0])

    return np.sum(variances)/df_top_clusters.shape[0]

list_tosave = []
for category in tqdm(list_categories):
#for category in ["telephone"]:
    
    # Get best distance parameter for dbscan       
    best_dist = get_best_dbcv(category)

    if best_dist is None:
        continue

    # Open dbscan file
    df_umap = pandas.read_csv(data_dir+f"{tag}{category}_dbscan_umap-r(1.6, 2, 9).csv")

    if len(df_umap) >= 2000000:
        alpha_par = 0.007
    else:
        alpha_par = 0.8

    # Remove outliers
    df_umap = remove_outliers(df_umap)

    # Get big clusters
    df_top_clusters, tot_n_clusters = get_big_clusters(df_umap, best_dist)
    n_big_clusters = len(df_top_clusters[f"DBSCAN (d={best_dist})"].unique())

    # Plots
    plot_umap(df_umap, alpha_par, category)
    plot_umap_clusters(df_top_clusters, alpha_par, category, best_dist)

    # Get % noise
    perc_noise = get_perc_noise(df_top_clusters, best_dist)

    # Compute weighted average of variance of umap points per cluster (weight is the number of points in the cluster)
    avg_var = compute_avg_variance(df_top_clusters, best_dist)

    list_tosave.append([category, n_big_clusters, tot_n_clusters, perc_noise, avg_var])

df_results = pd.DataFrame(list_tosave, columns=["category", "n_big_clusters", "tot_n_clusters", "perc_noise", "avg_var"])
# if folder "results_downstream" does not exist, create it
if not os.path.exists("../results/clusterability/"):
    os.makedirs("../results/clusterability/")
df_results.to_csv(f"../results/clusterability/dbscan_clusterability.csv", index=False)
