import pickle
import sys
import pandas as pd
from umap_forma import *
from tqdm import tqdm
import umap
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# open metadata
with open("../data/umap_files/all_categories_metadata_clusters_sample.pkl", "rb") as f:
    metadata = pickle.load(f)

# Open the embeddings tensor
with open("../data/umap_files/all_categories_embeddings_tensor_clusters_sample.pkl", "rb") as f:
    all_embeddings_tensor = pickle.load(f)
    
print("Embedding load completed.", flush=True)

# from metadata, extract unique categories
unique_categories = set([meta["category"] for meta in metadata])
print(len(unique_categories), "unique categories found.", flush=True)

# Convert emb_data to DataFrame for fast filtering
emb_data_df = pd.DataFrame([{
    "key_id": meta["key_id"],
    "category": meta["category"],
    "emb": emb
} for meta, emb in zip(metadata, all_embeddings_tensor)])

# generate a list of categories in the order they appear in the metadata
categories_metadata = emb_data_df["category"].unique().tolist()

print("Embeddings DataFrame created.", flush=True)

category_clust = pd.read_csv("../data/categories_clusterable.csv", names=["category"])
category_nonclust = pd.read_csv("../data/categories_nonclusterable.csv", names=["category"])
all_categories = pd.concat([category_clust, category_nonclust])["category"].tolist()

cluster_centroids_all = {}

for category in tqdm(categories_metadata, desc="Processing categories"):
    if category in category_clust["category"].values:
        cluster_file = pd.read_csv(f"../data/umap_files/{category}_umap_clusters.csv")
    else:
        cluster_file = pd.read_csv(f"../data/umap_files/non-clusterable/{category}_umap_clusters.csv")

    # Use a set for fast lookup
    valid_keys = set(cluster_file["key_id"])

    # Create a mapping from key_id to cluster
    cluster_map = dict(zip(cluster_file["key_id"], cluster_file["cluster"]))

    # Filter emb_data efficiently
    category_data = emb_data_df[
        (emb_data_df["category"] == category) & (emb_data_df["key_id"].isin(valid_keys))
    ].copy()

    # Add cluster assignment
    category_data["cluster"] = category_data["key_id"].map(cluster_map)

    # Group by cluster and compute centroids
    cluster_centroids = (
        category_data.groupby("cluster")["emb"]
        .apply(lambda embs: np.mean(np.stack(embs), axis=0))
        .to_dict()
    )

    # Format keys with category
    cluster_centroids_all[category] = {
        f"{category}_{cluster}": centroid for cluster, centroid in cluster_centroids.items()
    }

# Flatten cluster centroids
all_centroids = [
    {"id": cluster_id, "emb": centroid}
    for clusters in cluster_centroids_all.values()
    for cluster_id, centroid in clusters.items()
]

df_centroids = pd.DataFrame(all_centroids)

# Compute cosine similarity
emb_matrix = np.stack(df_centroids["emb"])
similarity_matrix = cosine_similarity(emb_matrix)

df_similarity = pd.DataFrame(
    similarity_matrix,
    index=df_centroids["id"],
    columns=df_centroids["id"]
)

# Order by category name alphabetically
df_similarity = df_similarity.sort_index(axis=0).sort_index(axis=1)

# Save result
with open("../data/image_similarity.pkl", "wb") as f:
    pickle.dump(df_similarity, f)




