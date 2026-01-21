import os
import pandas as pd
import ast
import matplotlib.pyplot as plt

category = "donut"

results_dir = "../plots_downstream/"+category+"/conn_component/perc_60/"

if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Load key_ids of drawings to plot
df_key_ids = pd.read_csv(f"../data/grid_conn_component_sampled_perc60.csv")
df_key_ids = df_key_ids[df_key_ids["category"]==category]

list_tosave = []
for i, row in df_key_ids.iterrows():
    key_id = row["key_id"]
    list_tosave.append(key_id)

print(f"Number of key_ids to plot: {len(list_tosave)}", flush=True)

# Iterate over csv files 
directory = f"../data/csv_files/{category}/{category}/"

for file in os.listdir(directory):
    if file.endswith(".csv"):
        df = pd.read_csv(directory+file, on_bad_lines='skip')
        df = df[df["key_id"].isin(list_tosave)]
        if len(df) == 0:
            continue
        for i, row in df.iterrows():
            key_id = row["key_id"]
            drawing = ast.literal_eval(row["drawing"])
            plt.figure(figsize=(5, 5))
            ax = plt.gca()
            ax.set_frame_on(False)  # Remove all spines
            ax.set_xticks([])  # Remove x ticks
            ax.set_yticks([])  # Remove y ticks
            
            for e in drawing:
                plt.plot(e[0], [-_ for _ in e[1]], 'k')  # Set all lines to black

            component = df_key_ids[df_key_ids["key_id"]==key_id]["connected_component"].values[0]
            
            plt.tight_layout()
            plt.savefig(results_dir+f"conn_component_{component}_{key_id}.png", dpi=300)
            plt.close()



