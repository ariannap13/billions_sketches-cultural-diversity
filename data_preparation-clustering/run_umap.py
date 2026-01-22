from umap_forma import *
import sys

category = sys.argv[1] # 'category'
H5_FILES = sys.argv[2]+category+"-embeddings_notime_fp16.h5" # 'H5_FILES dir'
CSV_FILES = sys.argv[3] # 'CSV_FILES dir'
SAVE_PATH = sys.argv[4] # 'SAVE_PATH dir'
BATCH_SIZE = int(sys.argv[5]) # 10000
SAVE_EVERY = int(sys.argv[6]) # 100000
emb_dir = sys.argv[10]

# make the save path if it does not exist
if not os.path.exists(SAVE_PATH):
    print('Creating save path', SAVE_PATH)
    os.makedirs(SAVE_PATH)

print('Category:', category, flush=True)

for file in os.listdir(emb_dir+"umap_notime/"):
    if file == category+"_umap_embeddings_dim_2":
        print(file, flush=True)
        print("File already exists")
        sys.exit(0)

GROUP_BY = 'countrycode'
# to make uniform, we sample from each country up to:
MAX_SAMPLES_PER_COUNTRY = int(sys.argv[7])

# how many of the countries (or lacales) with most entries to use
TOP_COUNTRIES = int(sys.argv[8])

### UMAP OPTIONS

UMAP_DIM = int(sys.argv[9]) # dimensions of UMAP
UMAP_EPOCHS = 500 # between 500-1000 is good

##### END OF OPTIONS
csv_file_list = []
for csv_file in os.listdir(CSV_FILES):
    if csv_file.endswith(".CSV") or csv_file.endswith(".csv"):
        csv_file_list.append(csv_file)
# extract the name of the category from the h5 file name iun CSV_FILES
csv_file_list = CSV_FILES
# take the first file and extract the name of the category
fnam = csv_file_list[0].split('_')
if len(fnam) > 2:
    category_name = fnam[0]+'_'+fnam[1]
else:
    category_name = fnam[0]

# sort the csv files by names
csv_file_list = sorted(csv_file_list)
# save_name_merged = os.path.join(SAVE_PATH, f'{category_name}_merged-{len(merged_df_info)}.csv')
save_name_merged = os.path.join(SAVE_PATH, f'{category}_merged.csv')
save_name_uni = os.path.join(SAVE_PATH, f'{category}_uniform-max{MAX_SAMPLES_PER_COUNTRY}-cnt{TOP_COUNTRIES}.csv')


# Uniformly sample from the embeddings based on country codes from the csv files
# This requires both the embedding H5 files as well as the original csv data files 

print('Loading data from CSV files to make uniform samples')
# check if the file exists. If yes, load it. If not, create merged_df_info and then save it
if os.path.exists(save_name_merged):
    print('Loading merged_df_info from', save_name_merged)
    merged_df_info = pd.read_csv(save_name_merged)
else:
    merged_df_info = get_merged_df(CSV_FILES+"*.csv")
    print('Saving merged_df_info to', save_name_merged)
    merged_df_info.to_csv(save_name_merged, index=False)

# uni_df 
# check if uniform sampling exists. If yes, load it. If not, create it and save it
if os.path.exists(save_name_uni):
    print('Loading uniform samples from', save_name_uni)
    uni_df = pd.read_csv(save_name_uni)
else:    
    print('Sampling uniformly from the data')
    uni_df = get_uniform_df(merged_df_info, GROUP_BY, 
                            top_column_values=TOP_COUNTRIES, 
                            num_samples=MAX_SAMPLES_PER_COUNTRY, 
                        )

    print('Saving to', save_name_uni)
    uni_df.to_csv(save_name_uni, index=False)

# UMAP

print('Starting UMAP')

clust = UmapEmbeddings(H5_FILES, uni_df, merged_df_info, num_pc=40)

# training
print('\n======\nTraining UMAP\n======\n')
clust.train_umap(epochs=UMAP_EPOCHS, dims=UMAP_DIM,)

# Embedding all data
print('\n======\nEmbedding all data\n======\n')
clust.embed_all_data(save_dir = SAVE_PATH, batch_size=BATCH_SIZE, save_every=SAVE_EVERY, umap_dims=UMAP_DIM)


