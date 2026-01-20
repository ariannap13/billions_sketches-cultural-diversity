import pickle
import sys
import pandas as pd
from umap_forma import *
from tqdm import tqdm
import umap
import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor


class PCA:
    def __init__(self, train_data, k=10):
        # Assuming 'train_data' is a PyTorch tensor with data on GPU
        # Center the data
        self.mean = train_data.mean(dim=0)
        centered_data = train_data - self.mean

        # Perform SVD
        u,s,vh = torch.linalg.svd(centered_data, full_matrices=False)

        # Take the first k principal components
        self.principal_components = vh[:, :k]
        
    def project(self, data):
        # Assuming 'data' is a PyTorch tensor with data on GPU
        centered_data = data - self.mean
        return centered_data @ self.principal_components
    

sample_max = 10000  # Maximum number of samples to take from each category

cat_clust = pd.read_csv("../data/categories_clusterable.csv", names=["category"])
cat_nonclust = pd.read_csv("../data/categories_nonclusterable.csv", names=["category"])

all_cat = pd.concat([cat_clust, cat_nonclust], axis=0)
all_cat = all_cat["category"].tolist()

# Function to load and process each category
def load_category_embeddings(category):

    # if the file does not exist, skip
    if not os.path.exists(f"../data/umap_files/{category}_embeddings_dict.pkl"):
        print(f"File for category {category} does not exist. Skipping.", flush=True)
        return []

    with open(f"../data/umap_files/{category}_embeddings_dict.pkl", "rb") as f:
        embeddings_df = pickle.load(f)  # This is a DataFrame

    if not isinstance(embeddings_df, pd.DataFrame):
        raise ValueError(f"Expected a DataFrame, but got {type(embeddings_df)} for category {category}", flush=True)
    
    if category in cat_clust["category"].values:
        cluster_file = pd.read_csv(f"../data/umap_files/{category}_umap_clusters.csv")
    elif category in cat_nonclust["category"].values:
        cluster_file = pd.read_csv(f"../data/umap_files/non-clusterable/{category}_umap_clusters.csv")
    
    set_tosamplefrom = cluster_file[cluster_file["cluster"]!=-1]

    # subset embeddings_df to only those clusters
    embeddings_df = embeddings_df[embeddings_df["key_id"].isin(set_tosamplefrom["key_id"].tolist())]

    #print(f"Loaded {len(embeddings_df)} embeddings for category {category}", flush=True)

    # Shuffle and sample (ensuring reproducibility)
    embeddings_sample = embeddings_df.sample(n=min(sample_max, len(embeddings_df)), random_state=42)

    return [
        (row["embeddings"], row["key_id"], category) 
        for _, row in embeddings_sample.iterrows()
    ]


# Parallel loading of embeddings
all_embeddings = []
metadata = []

with ThreadPoolExecutor(max_workers=4) as executor:  # Adjust `max_workers` based on CPU cores
    results = list(tqdm(executor.map(load_category_embeddings, all_cat), total=len(all_cat)))

# Filter out empty results (i.e. lists with length 0)
results = [result for result in results if result]

# Flatten results
all_embeddings = [emb for result in results for emb, _, _ in result]
metadata = [{"key_id": key_id, "category": category} for result in results for _, key_id, category in result]

# Convert embeddings to a tensor for faster processing
all_embeddings_tensor = torch.tensor(np.array(all_embeddings), dtype=torch.float32)

# save metadata
with open("../data/umap_files/all_categories_metadata_clusters_sample.pkl", "wb") as f:
    pickle.dump(metadata, f)

# Save the embeddings tensor
with open("../data/umap_files/all_categories_embeddings_tensor_clusters_sample.pkl", "wb") as f:
    pickle.dump(all_embeddings_tensor, f)

print("Embedding collection completed.", flush=True)
