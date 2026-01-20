import re
import pandas as pd

df_imagery = pd.read_csv("./imaginery_scores_llama3.3.csv")

def extract_rating(text):
    """
    Extracts the first integer found in the input string.
    
    Args:
        text (str): The input string containing a numerical rating.
        
    Returns:
        int or None: The extracted integer if found, else None.
    """
    match = re.search(r'\b\d+\b', text)
    return int(match.group()) if match else None

df_imagery["score"] = df_imagery["score"].apply(extract_rating)


df_imagery.to_csv("./imaginery_scores_llama3.3.csv", index=False)