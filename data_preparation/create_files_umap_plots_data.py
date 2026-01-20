import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import sys
import matplotlib.pyplot as plt
import seaborn as sns

data_dir = "../data/umap_files/"
tag = "2d_"
# category = "necklace"

# Define functions
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
    return df_top_clusters, tot_n_clusters

categories_clusterable = pd.read_csv("../data/categories_clusterable.csv", names=['category'])

# Main

for category in tqdm(categories_clusterable['category']):
    ## Get best distance parameter for dbscan       
    best_dist = get_best_dbcv(category)

    ## Open dbscan file
    df_umap = pd.read_csv(data_dir+f"{tag}{category}_dbscan_umap-r(1.6, 2, 9).csv")

    ## Remove outliers
    df_umap = remove_outliers(df_umap)

    ## Get big clusters
    df_top_clusters, tot_n_clusters = get_big_clusters(df_umap, best_dist)
    n_big_clusters = len(df_top_clusters[f"DBSCAN (d={best_dist})"].unique())

    # df_top_clusters = df_top_clusters[df_top_clusters[f"DBSCAN (d={best_dist})"] != -1]

    df_top_clusters_best = df_top_clusters[["key_id", "UMAP_1", "UMAP_2", f"DBSCAN (d={best_dist})"]]

    df_top_clusters_best.rename(columns={f"DBSCAN (d={best_dist})": "cluster"}, inplace=True)

    df_top_clusters_best.to_csv(f"../data/umap_files/{category}_umap_clusters.csv", index=False)
