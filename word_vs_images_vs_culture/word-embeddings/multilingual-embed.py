from sentence_transformers import SentenceTransformer
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Load the model
model = SentenceTransformer("Alibaba-NLP/gte-multilingual-base", trust_remote_code=True)

category_clust = pd.read_csv("../../data/categories_clusterable.csv", names=["category"])
category_nonclust = pd.read_csv("../../data/categories_nonclusterable.csv", names=["category"])

category_names = pd.concat([category_clust, category_nonclust], ignore_index=True)
category_names = category_names["category"].tolist()

print(f"Total categories loaded: {len(category_names)}")

embeddings = model.encode(category_names, normalize_embeddings=True)  # passage_embeddings.shape (2, 768)

# Compute similarity
similarity_matrix = cosine_similarity(embeddings)  # passage_embeddings.shape (2, 768)

# Create DataFrame
similarity_df = pd.DataFrame(similarity_matrix, index=category_names, columns=category_names)

# order rows and columns by category names alphabetically
similarity_df = similarity_df.reindex(sorted(similarity_df.columns), axis=1)
similarity_df = similarity_df.reindex(sorted(similarity_df.index), axis=0)

# Save the similarity DataFrame to a pickle file
with open("../../data/multilingual_similarity.pkl", "wb") as f:
    pickle.dump(similarity_df, f)