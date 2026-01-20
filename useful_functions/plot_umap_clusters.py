import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

category = "smiley_face"

def plot_umap_clusters(df_top_clusters, alpha_par, category):
    fig, ax = plt.subplots(figsize=(10,10))

    # define available colors
    hex_colors = sns.color_palette(["#EF476F", "#FFD166", "#06D6A0", "#118AB2", "#073B4C", 
                                    "#7209b7", "#ff6700"])

    # get unique clusters
    clusters = sorted(df_top_clusters['cluster'].unique())
    
    # map clusters to colors
    palette = {}
    for i, c in enumerate(clusters):
        if c == -1:
            palette[c] = "grey"  # special color for outliers
        else:
            palette[c] = hex_colors[i-1]  # cycle colors if not enough

    # plot
    sns.scatterplot(x="UMAP_1", y="UMAP_2", data=df_top_clusters, 
                    alpha=alpha_par, s=1, hue="cluster", palette=palette, ax=ax)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')

    # remove legend if category is not donut
    if category != "donut":
        ax.legend_.remove()

    # save
    plt.savefig(f"../plots_downstream/umap_top_clusters_{category}.png", transparent=True)
    plt.close()

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
    grid_size = 50  # Adjust this value as needed
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


umap_df = pd.read_csv(f"../data/umap_files/{category}_umap_clusters.csv")

# remove outliers (surrounded by empty space)
umap_df = remove_outliers(umap_df)

# plot umap
plot_umap_clusters(umap_df, alpha_par=0.5, category=category)