import pickle
import pandas as pd
from tqdm import tqdm
import os
import numpy as np

categories = pd.read_csv("../data/categories_nonclusterable.csv", names=["category"])

for category in tqdm(categories["category"], desc="Processing categories"):

    print(f"Processing category: {category}", flush=True)

    # cluster assignments
    df_top_clusters = pd.read_csv(f"../data/non-clusterable/{category}_umap_clusters.csv")

    # load embedding file containing countries
    with open(f"../data/umap_files/{category}_embeddings_dict.pkl", "rb") as f:
        country_embeddings = pickle.load(f)

    country_info = country_embeddings[["key_id", "countrycode"]].copy()

    # merge df_top_clusters with country_info
    df_top_clusters = df_top_clusters.merge(country_info, on="key_id", how="left")

    # get the counts of each cluster-country pair
    cluster_country_counts = df_top_clusters.groupby([f"cluster", "countrycode"]).size().unstack().fillna(0)
    # get the counts of each cluster
    cluster_counts = df_top_clusters["cluster"].value_counts()

    # compute the odds ratios
    odds_ratios = {}

    for country in cluster_country_counts.columns:
        or_country = []
        total_country = cluster_country_counts[country].sum()
        total_all = cluster_country_counts.sum().sum()
        total_other_countries = total_all - total_country

        for cluster in cluster_country_counts.index:
            if cluster == -1:
                continue

            a = cluster_country_counts.loc[cluster, country]
            b = total_country - a
            c = cluster_country_counts.loc[cluster].sum() - a
            d = total_other_countries - c

            # To avoid division by zero
            if b * c == 0:
                odds_ratio = float('inf')
            else:
                odds_ratio = (a * d) / (b * c)
            or_country.append(odds_ratio)

        odds_ratios[country] = or_country

    # similarity matrix of countries, use manhattan distance
    similarity_matrix = np.full((len(odds_ratios), len(odds_ratios)), np.nan)
    for i, country1 in enumerate(odds_ratios.keys()):
        for j, country2 in enumerate(odds_ratios.keys()):
            epsilon = 1e-10  # small positive value to avoid log(0)
            or1 = np.clip(odds_ratios[country1], epsilon, None)
            or2 = np.clip(odds_ratios[country2], epsilon, None)

            similarity_matrix[i, j] = np.linalg.norm(np.log(or1) - np.log(or2))

    # save similarity matrix
    # if the directory does not exist, create it
    if not os.path.exists(f"../results/similarity_matrices/non-clusterable"):
        os.makedirs(f"../results/similarity_matrices/non-clusterable") 
    np.save(f"../results/similarity_matrices/non-clusterable/{category}_similarity_matrix_log_euclidean.npy", similarity_matrix)

    with open(f"../results/similarity_matrices/non-clusterable/{category}_countries.pkl", "wb") as f:
        pickle.dump(odds_ratios, f)