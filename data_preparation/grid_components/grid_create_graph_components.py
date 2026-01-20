import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from igraph import Graph, plot
import warnings

warnings.filterwarnings("ignore")

data_dir = "../../data/umap_files/"
tag = "2d_"
clusterable = False
if clusterable:
    cl_tag = "_clust"
else:
    cl_tag = ""
# cluster_analysis = -1 # -1 when noise, otherwise choose cluster


# Function to extract grid coordinates from the 'grid_cell' column
def get_grid_coordinates(grid_cell):
    i, j = map(int, grid_cell.split(','))
    return i, j

# Function to get the index of a grid cell based on its (i, j) coordinates
def get_index(i, j, grid_width=20):
    return i * grid_width + j

if clusterable:
    categories_to_process = pd.read_csv("../../data/categories_files/categories_clusterable.csv", names=['category'])
else:
    categories_to_process = pd.read_csv("../../data/categories_files/categories_nonclusterable.csv", names=['category'])

all_grid_counts = []
for category in tqdm(categories_to_process['category']):
    # save grid cell
    df_top_clusters_best = pd.read_csv(f"../../results/continuum/grid_data/{category}_umap_grid{cl_tag}.csv")

    # Count number of points in each grid cell
    grid_counts = df_top_clusters_best['grid_cell'].value_counts().reset_index()
    grid_counts.columns = ['grid_cell', 'count']
    grid_counts["total_counts"] = grid_counts["count"].sum()
    grid_counts["perc_count"] = grid_counts["count"]/grid_counts["total_counts"]
    grid_counts["category"] = category

    all_grid_counts.append(grid_counts)

all_grid_counts = pd.concat(all_grid_counts)
# threshold = np.percentile(all_grid_counts["perc_count"], 65)

# print(f"Threshold: {threshold}", flush=True)


for category in tqdm(categories_to_process['category']):
    for threshold_perc in [60,70,80,90,95]:

        threshold = np.percentile(all_grid_counts["perc_count"], threshold_perc)
        print(f"Threshold perc: {threshold_perc}", flush=True)

        # save grid cell
        df_top_clusters_best = pd.read_csv(f"../../results/continuum/grid_data/{category}_umap_grid{cl_tag}.csv")

        df_top_clusters_best_copy = df_top_clusters_best.copy()

        # remove null grid cells
        df_top_clusters_best = df_top_clusters_best.dropna(subset=['grid_cell'])

        # select only those grid cells with more than 99th percentile of points
        # compute total counts and percentage of points in each grid cell
        df_top_clusters_best = df_top_clusters_best.merge(all_grid_counts[all_grid_counts["category"] == category], on="grid_cell")

        # select only those grid cells with more than 99th percentile of points
        df_top_clusters_best_thresholded = df_top_clusters_best[df_top_clusters_best['perc_count'] > threshold]

        unique_grid_cells = df_top_clusters_best_thresholded['grid_cell'].unique()

        # Create an empty adjacency matrix (20x20 grid, so 400 cells)
        grid_size = 20
        adj_matrix = np.zeros((400, 400))

        # generate all possible pairs "{i,j}" where i,j in [0, 19]
        combinations = [f"{i},{j}" for i in range(grid_size) for j in range(grid_size)]
        # set as indexes of the adjacency matrix and save for later mapping
        dict_combinations = dict(zip(combinations, range(400)))

        # Iterate through each unique grid cell
        for i in range(len(combinations)-1):
            for j in range(i + 1):  # Only compare once for each pair
                comb_i = combinations[i]
                comb_j = combinations[j]
                if comb_i in unique_grid_cells and comb_j in unique_grid_cells:
                    i1, j1 = get_grid_coordinates(comb_i)
                    i2, j2 = get_grid_coordinates(comb_j)
                    if abs(i1 - i2) <= 1 and abs(j1 - j2) <= 1:
                        adj_matrix[i, j] = adj_matrix[j, i] = 1
        
        # create graph from adjacency matrix, "highlight" the grid cells with more than 99th percentile of points
        g = Graph.Adjacency(adj_matrix.tolist(), mode="undirected")
        g.vs['label'] = combinations
        
        # g.vs['count'] = df_top_clusters_best['grid_cell'].value_counts().values
        # g.vs['color'] = ['red' if count > threshold else 'grey' for count in g.vs['count']]
        # g.vs['label'] = df_top_clusters_best['grid_cell'].value_counts().index

        # Assign UMAP coordinates to the vertices of the graph, use dict_combinations to map grid cells to indexes
        grid_cell_to_coordinates = dict(zip(df_top_clusters_best['grid_cell'], zip(df_top_clusters_best['UMAP_1'], df_top_clusters_best['UMAP_2'])))

        # if some of combinations are not in the unique grid cells, add them 
        for c in combinations:
            if c not in grid_cell_to_coordinates:
                grid_cell_to_coordinates[c] = (np.nan, np.nan)
        g.vs['x'] = [grid_cell_to_coordinates[cell][0] for cell in g.vs['label']]
        g.vs['y'] = [grid_cell_to_coordinates[cell][1] for cell in g.vs['label']]

        # Extract connected components (excluding single nodes)
        components = [c for c in g.connected_components() if len(c) > 1]

        # save the belonging of each grid cell to a connected component
        grid_cell_to_component = {}
        for i, component in enumerate(components):
            for vertex in component:
                grid_cell_to_component[g.vs[vertex]['label']] = i
        
        df_top_clusters_best['connected_component'] = df_top_clusters_best['grid_cell'].map(grid_cell_to_component)

        # plot umap with connected components
        plt.figure(figsize=(10, 10))
        plt.scatter(df_top_clusters_best_copy['UMAP_1'], df_top_clusters_best_copy['UMAP_2'], c="grey", alpha=0.01, s=10)

        # color all points based on the connected component, for points that are not in the connected component, color them as grey
        df_top_clusters_best['connected_component'] = df_top_clusters_best['connected_component'].fillna(-1)
        list_colors = sns.color_palette("hsv", len(components))
        for i in range(len(components)):
            df_connected_component = df_top_clusters_best[df_top_clusters_best['connected_component'] == i]
            color = list_colors[i]
            plt.scatter(df_connected_component['UMAP_1'], df_connected_component['UMAP_2'], label=f"Connected Component {i}", alpha=0.02, s=10, c=color)
        
        # fix color in legend to match the color of the connected component
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())

        grid_size = 20
        x_min, x_max = df_top_clusters_best_copy['UMAP_1'].min(), df_top_clusters_best_copy['UMAP_1'].max()
        y_min, y_max = df_top_clusters_best_copy['UMAP_2'].min(), df_top_clusters_best_copy['UMAP_2'].max()
        x_grid = np.linspace(x_min, x_max, grid_size)
        y_grid = np.linspace(y_min, y_max, grid_size)
        plt.xticks(x_grid, rotation=45)
        plt.yticks(y_grid)
        #plt.grid(True)
        # plot the 20x20 grid lines on umap plot

        # for i in range(20):
        #     plt.axvline(i, color='black', lw=0.5)
        #     plt.axhline(i, color='black', lw=0.5)

        # remove all spines
        ax = plt.gca()
        ax.set_frame_on(False)
        ax.set_xticks([])
        ax.set_yticks([])

        plt.savefig(f"../../plots_downstream/umap_grid/{category}_grid_connected_components{cl_tag}_threshold_perc_{threshold_perc}.png")
        plt.close()

        df_top_clusters_best.to_csv(f"../../results/continuum/grid_conn_component/{category}_umap_grid_connected_components{cl_tag}_threshold_perc_{threshold_perc}.csv", index=False)
