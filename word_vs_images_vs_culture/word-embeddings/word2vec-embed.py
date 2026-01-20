import pandas as pd
import nltk
import gensim.downloader
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import numpy as np

category_clust = pd.read_csv("../data/categories_clusterable.csv", names=["category"])
category_nonclust = pd.read_csv("../data/categories_nonclusterable.csv", names=["category"])

category_names = pd.concat([category_clust, category_nonclust], ignore_index=True)
category_names = category_names["category"].tolist()

print(f"Total categories loaded: {len(category_names)}")

word2vec_vectors = gensim.downloader.load('word2vec-google-news-300')

def get_word2vec_vector(word):
    try:
        vector = word2vec_vectors[word]
        # normalize the vector
        vector = vector / np.linalg.norm(vector)
    except KeyError:
        vector = None
    return vector

valid_categories = []
vectors = []

count_not_found = 0
for category in category_names:
    vector = get_word2vec_vector(category)
    if vector is not None:
        valid_categories.append(category)
        vectors.append(vector)
    else:
        count_not_found += 1
        print(f"Category '{category}' not found in Word2Vec model.")

print(f"Total categories found in Word2Vec model: {len(valid_categories)}")
print(f"Total categories not found in Word2Vec model: {count_not_found}")

# Compute similarity
similarity_matrix = cosine_similarity(vectors)

# Create DataFrame
similarity_df = pd.DataFrame(similarity_matrix, index=valid_categories, columns=valid_categories)

# add categories for which no vector was found with values set to NaN
for category in category_names:
    if category not in valid_categories:
        similarity_df[category] = np.nan
        similarity_df.loc[category] = np.nan

# order rows and columns by category names alphabetically
similarity_df = similarity_df.reindex(sorted(similarity_df.columns), axis=1)
similarity_df = similarity_df.reindex(sorted(similarity_df.index), axis=0)

# Save the similarity DataFrame to a pickle file
with open("../data/word2vec_similarity.pkl", "wb") as f:
    pickle.dump(similarity_df, f)