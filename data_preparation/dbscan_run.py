import numpy as np
import os
import sys
from forma import *
import time
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from collections import Counter
from scipy.spatial import distance
import numpy as np
from sklearn.metrics import pairwise_distances



# start timer
start = time.time()

# change according to needs
category = sys.argv[1] # 'category'
n_dims = int(sys.argv[2]) # 2
umap_emb_dir = sys.argv[3]+f'umap_notime/{category}_umap_embeddings_dim_{n_dims}/'
dbscan_save_dir = sys.argv[4]

## Functions

def dbscan_n(df, n_dims, distance_factor=2):
    """Perform DBSCAN clustering on UMAP results.
    distance_factor: determines DBSCAN's eps = distance_factor*(estimated volume of the data)
        Good range between 1.2-2.5 (larger means more clusters will merge)
    """

    list_umap_dims = ["UMAP_"+str(i+1) for i in range(n_dims)]
    # Extract UMAP coordinates from the DataFrame
    umap_coordinates = df[list_umap_dims].values
    
    # Compute the standard deviations along each dimension
    sigma = umap_coordinates.std(axis=0)
    
    # Compute the spread in each dimension (3*sigma)
    spread = 3 * sigma

    # Calculate the volume of the bounding box
    volume = np.prod(spread)

    # Calculate epsilon for DBSCAN
    eps = distance_factor * ((volume/umap_coordinates.shape[0]) ** (1/n_dims))  # Cube root for 3D volume
    
    # Instantiate and fit DBSCAN
    db = DBSCAN(eps=eps, min_samples=10)
    db_clust_fit = db.fit(umap_coordinates)
    
    # Count clusters
    clust_cnt = Counter(db_clust_fit.labels_)        
    print('# clusters ', len(clust_cnt))
    
    return db_clust_fit


## Main

# Load UMAP data
data_dir = umap_emb_dir

all_df = []
for file in os.listdir(data_dir):
    if file.endswith("_embeddings_merged_file.csv"):
        print(file, flush=True)
        df = pd.read_csv(data_dir+file)
        all_df.append(df)

df = pd.read_csv(data_dir+f"{category}_umap_train.csv")
all_df.append(df)
df_umap = pd.concat(all_df, ignore_index=True)
# remove duplicates
df_umap = df_umap.drop_duplicates(subset='key_id', keep='first')

print("Number of samples:", len(df_umap), flush=True)

# DBSCAN
df = pd.DataFrame()
df['key_id'] = df_umap['key_id']
for i in range(n_dims):
    df[f'UMAP_{i+1}'] = df_umap[f'UMAP_{i+1}']

dbscan_range = (1.6,2,9) # start, end, num

for r in np.linspace(*dbscan_range):

    db_clust_fit = dbscan_n(df, n_dims, distance_factor=r)

    df[f'DBSCAN (d={r:.2f})'] = db_clust_fit.labels_

file_name_dbscan = str(n_dims) + f'd_{category}_dbscan_umap-r{dbscan_range}.csv'

df.to_csv(dbscan_save_dir+file_name_dbscan)