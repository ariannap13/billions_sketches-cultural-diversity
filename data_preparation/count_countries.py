import sys
from glob import glob
import pandas as pd
import os

category = sys.argv[1]
csv_dir = sys.argv[2]

def get_merged_df(path_pattern):

    csv_files = sorted(glob(path_pattern))  # Sort the list of files

    print('Reading and processing CSV files...', flush=True)
    
    # Use a generator to read and filter files lazily
    def filtered_dataframes():
        for i, file in enumerate(csv_files):
            print(f'{i*100/len(csv_files):.1f}% {file}', end='\r', flush=True)
            try:
                # Read only the required columns
                df = pd.read_csv(file, usecols=['key_id', 'countrycode'])
                yield df
            except Exception as e:
                print(f"Error reading file {file}: {e}", flush=True)
                continue

    # Concatenate and remove duplicates in one step
    merged_df_info = pd.concat(filtered_dataframes(), ignore_index=True).drop_duplicates(subset='key_id', keep='first')

    print("\nFinished reading and processing files.", flush=True)
    return merged_df_info

if not os.path.exists(csv_dir+f'{category}/{category}/'):
    csv_files = csv_dir+f"{category}/_{category}/*.csv"
else:
    csv_files = csv_dir+f"{category}/{category}/*.csv"

data = get_merged_df(csv_files)

df_countries_count = data["countrycode"].value_counts()

df_countries_count.to_csv(f"../data/country_counts/{category}_country_counts.csv")