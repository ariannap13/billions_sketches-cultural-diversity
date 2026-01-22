import pandas as pd
import pickle as pkl
from sklearn.metrics.pairwise import cosine_similarity
import os

all_languages = []
for file in os.listdir("../../data/languages"):
    language = file.split("_")[1].split(".")[0]  # Extract language code from filename
    print(f"Processing language: {language}")
    all_languages.append(language)

# Compute the similarity for each pair of languages by category, then average the similarities
def compute_similarity(embeddings1, embeddings2, categories):
    similarity_matrix = cosine_similarity(embeddings1, embeddings2)
    # get the diagonal of the similarity matrix
    similarity_list = []
    for i, category in enumerate(categories):
        similarity_list.append({
            "category": category,
            "similarity": similarity_matrix[i][i]
        })
    return pd.DataFrame(similarity_list)

# Load embeddings for each language and compute similarities
similarity_dfs = {}
for i in range(len(all_languages)):
    for j in range(i + 1, len(all_languages)):
        lang1 = all_languages[i]
        lang2 = all_languages[j]
        
        with open(f"../../data/embeddings_{lang1}.pkl", "rb") as f:
            embeddings1 = pkl.load(f)
        with open(f"../../data/embeddings_{lang2}.pkl", "rb") as f:
            embeddings2 = pkl.load(f)
        
        categories = pd.read_csv("../../data/categories_clusterable.csv", names=["category"])["category"].tolist()
        # categories += pd.read_csv("../data/categories_nonclusterable.csv", names=["category"])["category"].tolist()
        
        similarity_df = compute_similarity(embeddings1, embeddings2, categories)
        # get the average similarity for the pair of languages
        avg_similarity = similarity_df["similarity"].mean()
        similarity_dfs[f"{lang1}_{lang2}"] = avg_similarity
        
# Save in a dataframe with lang1, lang2, and average similarity
similarity_df = pd.DataFrame(similarity_dfs.items(), columns=["language_pair", "average_similarity"])
similarity_df[["lang1", "lang2"]] = similarity_df["language_pair"].str.split("_", expand=True)
similarity_df.drop(columns=["language_pair"], inplace=True)
# Save the similarity DataFrame to a csv file
similarity_df.to_csv("../../data/multilingual_similarity_average.csv", index=False)
