import pandas as pd
from tqdm import tqdm

df = pd.read_csv('../../data/categories_nonclusterable.csv', names=['category'])

for category in tqdm(df['category']):
    
    df_comp = pd.read_csv(f"../../results/continuum/grid_conn_component/{category}_umap_grid_connected_components_threshold_perc_60.csv")
    df_comp = df_comp[["key_id", "UMAP_1", "UMAP_2", "connected_component"]]
    df_comp.rename(columns={"connected_component": "cluster"}, inplace=True)

    df_comp["cluster"] = df_comp["cluster"].astype(int)

    df_comp.to_csv(f"../../data/umap_files/non-clusterable/{category}_umap_clusters.csv", index=False)
    