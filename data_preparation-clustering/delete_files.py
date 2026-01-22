import os
import sys

category = sys.argv[1] # Category of drawings 
directory = sys.argv[2] # Directory containing the zip files 

for file in os.listdir(directory):
    if file==category or file==category+"-embeddings":
        print(file, flush=True)
        for subfile in os.listdir(directory+file+"/"+file):
            # if file is a directory, skip
            if os.path.isdir(directory+file+"/"+file+"/"+subfile):
                continue
            else:
                # delete the file
                os.remove(directory+file+"/"+file+"/"+subfile)

