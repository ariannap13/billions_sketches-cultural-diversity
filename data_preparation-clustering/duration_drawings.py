import numpy as np
import sys
import os
import pandas as pd
from glob import glob

category = sys.argv[1]
csv_files = sys.argv[2]

print(category, flush=True)

# create complexity folder if it does not exist
if not os.path.exists('../data/complexity'):
    os.makedirs('../data/complexity')

def get_merged_df(path_pattern):

    csv_files = sorted(glob(path_pattern))  # Sort the list of files

    print('Reading and processing CSV files...', flush=True)

    # Use a generator to read and filter files lazily
    def filtered_dataframes():
        for i, file in enumerate(csv_files):
            print(f'{i*100/len(csv_files):.1f}% {file}', end='\r', flush=True)
            try:
                # Read only the required columns
                df = pd.read_csv(file, usecols=['key_id', 'countrycode', 'locale', 'duration', 'drawing'])
                yield df
            except Exception as e:
                print(f"Error reading file {file}: {e}", flush=True)
                continue

    # Concatenate and remove duplicates in one step
    merged_df_info = pd.concat(filtered_dataframes(), ignore_index=True).drop_duplicates(subset='key_id', keep='first')

    print("\nFinished reading and processing files.", flush=True)
    return merged_df_info

if __name__ == '__main__':
    
    output_files = [
        f'../data/complexity/{category}_entropies.npy',
        f'../data/complexity/{category}_perc_colored_pixels.npy',
        f'../data/complexity/{category}_n_lines.npy',
        f'../data/complexity/{category}_duration.npy'
    ]
    
    if all(os.path.exists(file) for file in output_files):
        print(f'All files for {category} already exist. Skipping...', flush=True)
        sys.exit(0)

    if not os.path.exists(csv_files+f'{category}/{category}/'):
        csv_files = csv_files+f"{category}/_{category}/*.csv"
    else:
        csv_files = csv_files+f"{category}/{category}/*.csv"
    #csv_files = f"../data/{category}/*.csv"

    if category == "aircraft_carrier":
        csv_files = csv_files+f"{category}/{category}/*.CSV"

    data = get_merged_df(csv_files)

    # get size
    print(f"Number of rows: {len(data)}", flush=True)

    # if duration file exists, skip
    if os.path.exists(f'../data/complexity/{category}_duration.npy'):
        print(f'Duration file for {category} already exists. Skipping...', flush=True)
    else:
        duration = list(data['duration'].values)
        # save the entropies to a file
        np.save(f'../data/complexity/{category}_duration.npy', duration)
        print(f'Duration saved to {category}_duration.npy', flush=True)
