import zipfile
import os
import sys
import shutil

category = sys.argv[1] # Category of drawings (e.g. "cat")
directory = sys.argv[2] # Directory containing the zip files (e.g. "emb_csv_round6")

for file in os.listdir(directory):
    if file==category+".zip" or file==category+"-embeddings.zip":
        print(file, flush=True)
        if file.endswith(".zip"):
            zip_file_path = directory+file

            extract_dir = directory + file.split(".")[0]
            print(extract_dir, flush=True)

            # Open the zip file
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                # Extract all files
                zip_ref.extractall(extract_dir)

            # if directory already exists, rename it
            if os.path.exists(extract_dir+f"/_{category}/"):
                os.rename(extract_dir+f"/_{category}/", extract_dir+f"/{category}/") 

            # # change name of the zipped file
            # os.rename(zip_file_path, zip_file_path.split(".zip")[0]+"_extracted.zip")
