import pandas as pd
import numpy as np
import warnings
import os

warnings.filterwarnings("ignore")

dimensions = "all"
top_countries = 100

def create_dist_matrix(csv_file, mapping_df, start_year, matrix):
    df = pd.read_csv(csv_file)

    # Clean up column names
    list_culture_countries = [col.split(start_year)[0].strip() for col in df.columns[1:]]
    df.columns = ["country"] + list_culture_countries

    # Build mapping dict
    mapping_dict = dict(zip(mapping_df["culture_country"], mapping_df["code"]))

    # Map column names
    df.columns = ["country"] + [mapping_dict.get(x, x) for x in list_culture_countries]

    # Map row country names
    df["country"] = df["country"].map(
        lambda x: mapping_dict.get(x.split(start_year)[0].strip(),
                                   x.split(start_year)[0].strip())
    )

    # Set index
    df.set_index("country", inplace=True)

    # Align df to the same structure as matrix
    df = df.reindex(index=matrix.index, columns=matrix.columns)

    # Populate matrix with values from df (NaNs preserved otherwise)
    matrix.update(df)

    return matrix

# Countries we have in our data
top_100_countries = pd.read_csv("../../data/country_counts/table_counts.csv")
unique_countries = top_100_countries["country"].str.upper()[:top_countries]

# Create base matrix filled with NaN
matrix = pd.DataFrame(np.nan, index=unique_countries, columns=unique_countries)

data_country_map = pd.read_csv("../../data/countries-codes.csv")[[
    'OFFICIAL LANG CODE', 'ISO2 CODE', 'ISO3 CODE', 'ONU CODE',
    'IS ILOMEMBER', 'IS RECEIVING QUEST', 'LABEL EN'
]]

mapping_df = pd.read_csv("../../data/culture_country_mapping_checked.csv")
mapping_df["code"] = mapping_df["best_match"].map(
    lambda x: data_country_map.loc[data_country_map["LABEL EN"] == x, "ISO2 CODE"].values[0]
)


for year_range in ["2010-2014", "2005-2009", "1999-2004", "1994-1998"]:

    print(f"Processing year range: {year_range}")

    csv_file = f"./data/{dimensions}/{dimensions}_dimensions_{year_range}.csv"
    start_year = csv_file.split(f"dimensions_")[1].split("-")[0]

    # Build distance matrix by populating values
    dist_matrix = create_dist_matrix(csv_file, mapping_df, start_year, matrix)

    # for how many countries do we have data?
    n_countries_with_data = (dist_matrix.notna().sum(axis=1) > 0).sum()
    print(f"Number of countries with data: {n_countries_with_data} out of {len(unique_countries)}")

    dist_matrix.to_csv(f"./data/{dimensions}/{dimensions}_dist_matrix_{start_year}_top{top_countries}.csv")

