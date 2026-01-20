import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import pandas as pd
import networkx as nx
import numpy as np
from scipy import integrate
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import sys

clusterable = True
non_clusterable = True
top_countries_n = int(sys.argv[1])

if clusterable and not non_clusterable:
    clusterable_tag = "clusterable"
elif clusterable and non_clusterable:
    clusterable_tag = "all"
elif not clusterable and non_clusterable:
    clusterable_tag = "non-clusterable"
             
    
def standardized_log_odds(odds):
    odds = np.array(odds)
    odds = np.where(odds == 0, 1e-10, odds)  # Prevent log(0) errors
    log_odds = np.log(odds)
    mean_log_odds = np.mean(log_odds)
    std_log_odds = np.std(log_odds)
    if std_log_odds == 0:  # Prevent division by zero
        std_log_odds = 1e-10  # Assign zero vector instead
    return (log_odds - mean_log_odds) / std_log_odds

def similarity_standardized_log_odds(odds1, odds2):
    transformed_odds1 = standardized_log_odds(odds1)
    transformed_odds2 = standardized_log_odds(odds2)
    return cosine_similarity(transformed_odds1.reshape(1, -1), transformed_odds2.reshape(1, -1))[0, 0]

def similarity_euclidean(odds1, odds2):
    if len(odds1) != len(odds2):
        raise ValueError("The two vectors must have the same length.")
    elif len(odds1) == 0 and len(odds2) == 0:
        return 0
    
    # if all values are within 0.9 and 1.1 for both odds, return 0
    if all(0.9 <= x <= 1.1 for x in odds1) and all(0.9 <= x <= 1.1 for x in odds2):
        return 0
    
    transformed_odds1 = standardized_log_odds(odds1)
    transformed_odds2 = standardized_log_odds(odds2)

    # # Ensure no NaNs
    # transformed_odds1 = np.nan_to_num(transformed_odds1, nan=0.0)
    # transformed_odds2 = np.nan_to_num(transformed_odds2, nan=0.0)

    # Compute Euclidean distance
    dist = np.linalg.norm(transformed_odds1 - transformed_odds2)

    # Normalize to [0,1] using a sigmoid-like scaling
    similarity = 1 - (dist / (dist + 1))

    return similarity


# Main

if clusterable_tag != "all":

    categories = pd.read_csv(f"../data/categories_{clusterable_tag}.csv", names=["category"])["category"].tolist()
    for file in os.listdir(f"../results/similarity_matrices/{clusterable_tag}/"):
        if file.endswith("log_euclidean.npy"):
            category = file.split("_similarity_matrix_log_euclidean.npy")[0]

            # load country names
            with open(f"../results/similarity_matrices/{clusterable_tag}/"+f"{category}_countries.pkl", "rb") as f:
                countries_or = pickle.load(f)

            if type(countries_or) == dict:
                countries = list(countries_or.keys())
                sim_matrix = np.full((len(countries), len(countries)), np.nan)
                for i in range(len(countries)-1):
                    for j in range(i+1, len(countries)):
                        sim_matrix[i, j] = similarity_euclidean(countries_or[countries[i]], countries_or[countries[j]])
                        sim_matrix[j, i] = sim_matrix[i, j]
            else:
                continue

            # transform data to similarity
            data = sim_matrix

            sims = []
            for i in range(len(countries)):
                for j in range(i+1, len(countries)):
                    sims.append([countries[i], countries[j], float(data[i,j])])


            # dataframe
            sims_df = pd.DataFrame(sims, columns=["country1", "country2", "sim"])
            
            # save the dataframe
            sims_df.to_csv(f"../results/similarity_matrices/{clusterable_tag}/"+f"{category}_new_sim_1.csv", index=False)

# # keep only top 50 countries
# with open("../data/categories_files/top_50_countries_ML.pkl", "rb") as f:
#     top_countries = pickle.load(f)
top_countries = pd.read_csv("../data/country_counts/table_counts.csv")
top_countries = list(top_countries["country"])[:top_countries_n]

if clusterable_tag != "all":
    average_distances = {}
    for file in os.listdir(f"../results/similarity_matrices/{clusterable_tag}/"):
        #if file.endswith("_sim.csv") and "clusterable" not in file:
        if file.endswith("_new_sim_1.csv"):
            category = file.split("_new_sim_1.csv")[0]

            data = pd.read_csv(f"../results/similarity_matrices/{clusterable_tag}/" + file)

            # for each row
            for i, row in data.iterrows():
                if row["country1"]+"-"+row["country2"] not in average_distances:
                    average_distances[row["country1"]+"-"+row["country2"]] = [row["sim"]]
                else:
                    average_distances[row["country1"]+"-"+row["country2"]].append(row["sim"])

    average_sims_list = []
    for key, value in average_distances.items():
        country1, country2 = key.split("-")
        average_sims_list.append([country1, country2, np.median(value)])

    average_sims_df = pd.DataFrame(average_sims_list, columns=["country1", "country2", "sim"])

    average_sims_df = average_sims_df[(average_sims_df["country1"].isin(top_countries)) & (average_sims_df["country2"].isin(top_countries))]

    # save the dataframe
    average_sims_df.to_csv(f"../results/similarity_matrices/{clusterable_tag}/"+"average_sims_top_countries.csv", index=False)

else:
    average_distances = {}
    clusterable_path = "../results/similarity_matrices/clusterable/"
    non_clusterable_path = "../results/similarity_matrices/non-clusterable/"

    for file in os.listdir(clusterable_path):
        #if file.endswith("_sim.csv") and "clusterable" not in file:
        if file.endswith("_new_sim_1.csv"):
            category = file.split("_new_sim_1.csv")[0]

            data = pd.read_csv(f"../results/similarity_matrices/clusterable/" + file)

            # for each row
            for i, row in data.iterrows():
                if row["country1"]+"-"+row["country2"] not in average_distances:
                    average_distances[row["country1"]+"-"+row["country2"]] = [row["sim"]]
                else:
                    average_distances[row["country1"]+"-"+row["country2"]].append(row["sim"])
    
    for file in os.listdir(non_clusterable_path):
        #if file.endswith("_sim.csv") and "clusterable" not in file:
        if file.endswith("_new_sim_1.csv"):
            category = file.split("_new_sim_1.csv")[0]
            
            data = pd.read_csv(f"../results/similarity_matrices/non-clusterable/" + file)

            # for each row
            for i, row in data.iterrows():
                if row["country1"]+"-"+row["country2"] not in average_distances:
                    average_distances[row["country1"]+"-"+row["country2"]] = [row["sim"]]
                else:
                    average_distances[row["country1"]+"-"+row["country2"]].append(row["sim"])


    average_sims_list = []
    for key, value in average_distances.items():
        country1, country2 = key.split("-")
        average_sims_list.append([country1, country2, np.median(value)])

    average_sims_df = pd.DataFrame(average_sims_list, columns=["country1", "country2", "sim"])

    average_sims_df = average_sims_df[(average_sims_df["country1"].isin(top_countries)) & (average_sims_df["country2"].isin(top_countries))]

    # save the dataframe
    average_sims_df.to_csv(f"../results/similarity_matrices/all_average_sims_top_{top_countries_n}_countries.csv", index=False)

