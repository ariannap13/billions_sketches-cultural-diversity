# open h5 file and merge the embeddings from folder "results"

import os
import sys
from forma import *
import sys

category = sys.argv[1]
emb_dir = sys.argv[2]

# findall files of format .h5 in category+"-embeddings" folder
files_tot_16 = []
files_tot_32 = []
for file in os.listdir(emb_dir+f"{category}-embeddings/{category}-embeddings/"):
    if category+"-embeddings_notime_fp16.h5" in file:
        print("File already exists")
        sys.exit(0)
    # ends with h5 and contains fp16
    if file.endswith(".h5") and "fp16" in file:
        files_tot_16.append(file)
    elif file.endswith(".h5") and "fp32" in file and "new" not in file:
        files_tot_32.append(file)

files_tot_16.sort()
files_tot_32.sort()

# merge the embeddings in 
first=True
for file in files_tot_16:
    print(file, flush=True)
    # load using load_embedding_h5
    try:
        embeddings_dict = load_embedding_h5(emb_dir+f"{category}-embeddings/{category}-embeddings/"+file)
        # get the embeddings and ids
        embeddings = embeddings_dict['embeddings']
        ids = embeddings_dict['key_id']
        # append to the total embeddings and ids
        if first:
            embeddings_tot = embeddings
            ids_tot = ids
            first=False
        else:
            embeddings_tot = torch.vstack((embeddings_tot, embeddings))
            ids_tot.extend(ids)
    except:
        print("corrupted file", flush=True)

if len(files_tot_16)>0:
     # save merged embeddings
     # if file does not exist
     if not os.path.exists(emb_dir+f"{category}-embeddings/{category}-embeddings/"+category+"-embeddings_notime_fp16.h5"):
         save_h5(emb_dir+f"{category}-embeddings/{category}-embeddings/{category}-embeddings_notime_fp16.h5", category, ids_tot, embeddings_tot)
