import pandas as pd
import os 
import ast

df = pd.read_csv("../../data/country_counts/table_counts.csv")
df["country"] = df["country"].str.lower()

df_lang = pd.read_csv("../../data/country_annotation.csv")
# Extract first language
# Convert string to list
df_lang['languages_list'] = df_lang['languages'].apply(ast.literal_eval)
# Extract first language
df_lang['first_language'] = df_lang['languages_list'].str[0]

language_codes = pd.read_csv("../../data/language-codes.csv")
df_lang = df_lang.merge(language_codes, right_on="English", left_on="first_language", how="left")

df = df.merge(df_lang[["alpha2", "code"]], left_on="country", right_on="code", how="left")

# Create final language column
df['final_language'] = df['alpha2'].fillna(df['language'])

df.to_csv("../../data/country_counts/table_counts.csv")

unique_languages = set(df["final_language"].values)
print(f"Unique languages: {len(unique_languages)}")

for lang in unique_languages:
    if not os.path.exists(f"../../data/languages/categories_{lang}.csv"):
        # create file
        df = pd.DataFrame(columns=["category"])
        df.to_csv(f"../../data/languages/categories_{lang}.csv", index=False)

