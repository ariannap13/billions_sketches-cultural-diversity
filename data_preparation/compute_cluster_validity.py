import pandas as pd
import numpy as np
import dbcv
import sys

# ignore warnings
import warnings
warnings.filterwarnings("ignore")

# time the code
import time

category = sys.argv[1]

seed = int(sys.argv[2])
ndim = int(sys.argv[3])
dbscan_save_dir = sys.argv[4]
emb_dir = sys.argv[5]


n_seed = [seed+i for i in range(5)]

file_dbscan = dbscan_save_dir+str(ndim)+"d_" + category + "_dbscan_umap-r(1.6, 2, 9).csv"
file_umap = emb_dir+f"umap_notime/{category}_umap_embeddings_dim_{ndim}/{category}_umap_train.csv"

df_dbscan = pd.read_csv(file_dbscan)
df_umap_train = pd.read_csv(file_umap)

# in df_dbscan, select only the rows with key_id in df_umap_train
df = df_dbscan[df_dbscan["key_id"].isin(df_umap_train["key_id"])]

print("Number of initial samples: ", len(df))

all_runs = []
#for n_samples in [500, 1000, 2000, 5000, 10000]:
for n_samples in [1000]:
    start = time.time()
    for d in ["1.60", "1.65", "1.70", "1.75", "1.80", "1.85", "1.90", "1.95", "2.00"]:
        print("n_samples = ", n_samples, " d = ", d, flush=True)
        col_name = f"DBSCAN (d={d})"

        for run in n_seed:
            # sample size based on dataframe length
            sample_size = min(len(df), n_samples)
            # sample dataframe
            df_sampled = df.sample(n=sample_size, random_state=run)
            # calculate validity index
            list_umap_dims = ["UMAP_"+str(i+1) for i in range(ndim)]
            features_sampled = np.array(df_sampled[list_umap_dims])
            labels_sampled = np.array(df_sampled[col_name])
            dbcv_metric = dbcv.dbcv(features_sampled, labels_sampled, check_duplicates=False)
            all_runs.append([n_samples, d, seed, dbcv_metric])

    end = time.time()
    # print time in minutes
    print("Time for n_samples = ", n_samples, " is ", (end - start)/60, " minutes", flush=True)

# save to csv
df_overall = pd.DataFrame(all_runs, columns=["n_samples", "d", "seed", "dbcv_metric"])
df_overall.to_csv(dbscan_save_dir+str(ndim)+"d_dbscan_dbcv_study_"+category+"_train_startseed_"+str(seed)+".csv", index=False)


