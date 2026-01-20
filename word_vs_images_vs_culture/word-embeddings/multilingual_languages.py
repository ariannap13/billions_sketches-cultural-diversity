from sentence_transformers import SentenceTransformer
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
from tqdm import tqdm

all_languages = []
for file in os.listdir("../data/languages"):
    language = file.split("_")[1].split(".")[0]  # Extract language code from filename
    all_languages.append(language)

# Load the model
model = SentenceTransformer("Alibaba-NLP/gte-multilingual-base", trust_remote_code=True)

for language in tqdm(all_languages):
    category = pd.read_csv(f"../data/languages/categories_{language}.csv")
    # drop any row that is empty or contains only whitespace
    category["category"] = category["category"].str.strip()
    category = category[category["category"] != ""]  # Remove empty strings
    category = category[category["category"] != " "]  # Remove empty strings
    category.dropna(inplace=True)  # Drop any rows with NaN values

    # categories_clusterable = pd.read_csv("../data/categories_clusterable.csv", names=["category"])
    # categories_nonclusterable = pd.read_csv("../data/categories_nonclusterable.csv", names=["category"])
    # category = pd.concat([categories_clusterable, categories_nonclusterable], ignore_index=True)
    category = [x.lower() for x in category["category"].tolist()]
    embeddings = model.encode(category, normalize_embeddings=True)  # passage_embeddings.shape (2, 768)

    # save embeddings to a pickle file
    with open(f"../data/embeddings_{language}.pkl", "wb") as f:
        pickle.dump(embeddings, f)