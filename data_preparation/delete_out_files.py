import os
import sys

category = sys.argv[1] # Category of drawings (e.g. "cat")

for file in os.listdir("./"):
    if (file.endswith(".out") and category in file) or (file.startswith("slurm") and file.endswith(".out")):
        os.remove(file)
        print("Deleted", file, flush=True)