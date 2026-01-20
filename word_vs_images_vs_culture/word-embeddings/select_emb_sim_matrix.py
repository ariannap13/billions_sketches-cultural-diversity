import os
import pandas as pd

top_countries = 100

# open top countries dataset
countries_df = pd.read_csv("../data/country_counts/table_counts.csv").head(top_countries)

similarity_df = pd.read_csv("../data/multilingual_similarity_average.csv")

# make a set for faster lookup
valid_langs = set(countries_df["final_language"])

# filter rows where both languages are in top countries
new_sim_df = similarity_df[
    similarity_df["lang1"].isin(valid_langs) & similarity_df["lang2"].isin(valid_langs)
].reset_index(drop=True)

new_sim_df.to_csv(f"../data/multilingual_similarity_average_top{top_countries}.csv")