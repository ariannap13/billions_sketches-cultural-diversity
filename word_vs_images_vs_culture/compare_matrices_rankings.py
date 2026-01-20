from email.mime import image
from operator import mul
import pickle
import pandas as pd
import rbo
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np
from scipy import stats
from collections import defaultdict


word2vec_similarity = pd.read_pickle("../data/word2vec_similarity.pkl")
multilingual_similarity = pd.read_pickle("../data/multilingual_similarity.pkl")
image_similarity = pd.read_pickle("../data/image_similarity.pkl")

def create_dataframe(sim_dict):
    agg_dict = {}
    for category, variants in sim_dict.items():
        rbo_5_list = []
        rbo_10_list = []
        rbo_20_list = []
        rbo_5_baseline_list = []
        rbo_10_baseline_list = []
        rbo_20_baseline_list = []
        kendall_tau_list = []
        kendall_tau_baseline_list = []
        overlap_top10_list = []
        overlap_top10_baseline_list = []

        if len(variants) == 0:
            continue
        
        for variant in variants.values():
            rbo_5_list.append(variant.get("rbo_5", 0))
            rbo_10_list.append(variant.get("rbo_10", 0))
            rbo_20_list.append(variant.get("rbo_20", 0))
            rbo_5_baseline_list.append(variant.get("rbo_5_baseline", 0))
            rbo_10_baseline_list.append(variant.get("rbo_10_baseline", 0))
            rbo_20_baseline_list.append(variant.get("rbo_20_baseline", 0))
            kendall_tau_list.append(variant.get("kendall_tau", 0)) 
            kendall_tau_baseline_list.append(variant.get("kendall_tau_baseline", 0))         
            overlap_top10_list.append(variant.get("overlap_top10", 0))
            overlap_top10_baseline_list.append(variant.get("overlap_top10_baseline", 0))


        agg_dict[category] = {
            "RBO@5": sum(rbo_5_list)/len(rbo_5_list),
            "RBO@10": sum(rbo_10_list)/len(rbo_10_list),
            "RBO@20": sum(rbo_20_list)/len(rbo_20_list),
            "Baseline@5": sum(rbo_5_baseline_list)/len(rbo_5_baseline_list),
            "Baseline@10": sum(rbo_10_baseline_list)/len(rbo_10_baseline_list),
            "Baseline@20": sum(rbo_20_baseline_list)/len(rbo_20_baseline_list),
            "kendall_tau": sum(kendall_tau_list) / len(kendall_tau_list),   
            "kendall_tau_baseline": sum(kendall_tau_baseline_list) / len(kendall_tau_list),       
            "overlap_top10": sum(overlap_top10_list) / len(overlap_top10_list),
            "overlap_top10_baseline": sum(overlap_top10_baseline_list) / len(overlap_top10_baseline_list),
        }

    df = pd.DataFrame(agg_dict).T
    return df

def plot_ranking_comparisons(df_aggregated, embedding_model):
    df_plot = []

    for k in [5, 10, 20]:
        df_plot.append(pd.DataFrame({
            "K": f"top-{k}",
            "Type": "Words",
            "Value": df_aggregated[f"RBO@{k}"].values
        }))
        df_plot.append(pd.DataFrame({
            "K": f"top-{k}",
            "Type": "Baseline",
            "Value": df_aggregated[f"Baseline@{k}"].values
        }))

    df_plot = pd.concat(df_plot, ignore_index=True)

    # Plot side-by-side boxplots
    plt.figure(figsize=(5,5))
    sns.boxplot(data=df_plot, x='K', y='Value', hue='Type', palette=['skyblue','lightcoral'], showfliers=False)
    plt.title("")
    plt.ylabel("Similarity of rankings\n(Ranked-Biased Overlap)")
    plt.xlabel("")
    plt.legend(title="")
    sns.despine()
    plt.ylim((-0.2, 1))
    plt.tight_layout(pad=2)  # more padding
    plt.savefig(f"../plots_downstream/rbo_{embedding_model}_image_baseline_sidebyside.pdf", dpi=300)
    plt.close()

    # Kendall's Tau scores (w2v vs multi, w2v vs image, multi vs image)
    ## Boxplot for Kendall's Tau scores
    plt.figure(figsize=(5, 5))
    sns.boxplot(data=[df_aggregated["kendall_tau"], df_aggregated["kendall_tau_baseline"]], orient='v', showfliers=False, palette=['skyblue','lightcoral'])
    plt.title("")
    plt.xticks([0, 1], ['Words', 'Baseline'])
    plt.ylabel("Similarity of rankings\n(Kendall-Tau on full rank)")
    sns.despine()
    plt.ylim((-0.2, 1))
    plt.tight_layout(pad=2)  # more padding
    plt.savefig(f"../plots_downstream/kendall_tau_scores_{embedding_model}.pdf", dpi=300)
    plt.close()

    # Overalap of top-10 categories percentage
    ##  Boxplot for overlap of top-10 categories
    plt.figure(figsize=(5, 5))
    sns.boxplot(data=[df_aggregated["overlap_top10"], df_aggregated["overlap_top10_baseline"]], orient='v', showfliers=False, palette=['skyblue','lightcoral'])
    plt.title("")
    plt.xticks([0, 1], ['Words', 'Baseline'])
    plt.ylabel("Similarity of rankings\n(overlap in top-10)")
    plt.ylim((-0.2, 1))
    plt.tight_layout(pad=2)  # more padding
    sns.despine()
    plt.savefig(f"../plots_downstream/overlap_top10_scores_{embedding_model}.pdf", dpi=300)

# global mixing for images
tot = 0
tot_first = 0
tot_top5 = 0
first_ranks_list = []
ranks_list = []
ranks_list_noduplicates = []

for category in image_similarity.index:
    clean_category = "_".join(category.split("_")[:-1])
    
    # Get the similarity row
    image_row = image_similarity.loc[category].copy()
    
    # Remove self-match
    image_row = image_row.drop(category)
    
    # Rank descending
    rank_row = image_row.rank(ascending=False)
    
    # Convert to DataFrame
    image_rank_df = rank_row.to_frame(name='rank')
    
    # Find indices with the same category
    same_category_indices = [idx for idx in image_rank_df.index if clean_category + "_" in idx]
    
    if not same_category_indices:
        continue
    
    # Rank of first same-category match
    first_rank = image_rank_df.loc[same_category_indices[0], "rank"]
    ranks_list.append(first_rank/len(image_rank_df))  # normalized rank
    first_ranks_list.append(first_rank)

    # Add cleaned category column
    image_rank_df['category_cleaned'] = (
        image_rank_df.index.str.split("_").str[:-1].str.join("_")
    )

    # Sort by original similarity rank
    image_rank_df = image_rank_df.sort_values('rank')

    # Keep only one representative per cleaned category (closest one)
    image_rank_df = image_rank_df.drop_duplicates(
        subset=['category_cleaned'], keep='first'
    )

    # Recompute ranking
    image_rank_df['rank_after'] = (
        image_rank_df['rank'].rank(method='first').astype(int)
    )

    # Recompute same-category indices NOW that duplicates are removed
    same_category_indices = image_rank_df[
        image_rank_df["category_cleaned"] == clean_category
    ].index

    if len(same_category_indices) > 0:
        # Absolute normalized rank AFTER deduplication
        first_rank_after = image_rank_df.loc[same_category_indices[0], "rank_after"]
        normalized_rank_after = first_rank_after / len(image_rank_df)

        # store new normalized rank
        ranks_list_noduplicates.append(normalized_rank_after)

    if clean_category == "pizza":
        print("Image category:", category)

        top10 = image_rank_df.nsmallest(10, 'rank_after')
        print("Top-10 closest matches:", top10)

        if len(same_category_indices) > 0:
            first_rank_pizza = image_rank_df.loc[same_category_indices[0], "rank_after"]
            print("Pizza first rank:", first_rank_pizza)
        else:
            print("No pizza match found after deduplication.")
    # # clean image_rank_df to keep only one entry per category_cleaned (i.e., one cluster per category)
    # image_rank_df['category_cleaned'] = image_rank_df.index.str.split("_").str[:-1].str.join("_")   
    # # sort image_rank_df by rank
    # image_rank_df = image_rank_df.sort_values('rank')
    # image_rank_df = image_rank_df.drop_duplicates(subset=['category_cleaned'])
    # # re-assign rank after dropping duplicates
    # image_rank_df['rank_after'] = range(1, len(image_rank_df) + 1)

    # if clean_category == "pizza":
    #     print("Image category:", category)
    #     # print the top-10 closest matches
    #     top10 = image_rank_df.nsmallest(10, 'rank_after')
    #     first_rank_pizza = image_rank_df.loc[same_category_indices[0], "rank_after"]
    #     print("Top-10 closest matches:", top10)
    #     print("Pizza first rank:", first_rank_pizza)
    
    if first_rank == 1.0:
        tot_first += 1  # First non-self match is top-ranked
    elif first_rank <= 5.0:
        tot_top5 +=1
    tot += 1

print("Fraction of times first non-self match is top-ranked:", tot_first / tot)
print("Fraction of times first non-self match is ranked in top-5:", tot_top5 / tot)
print("Average rank of first non-self match:", np.mean(ranks_list))
print("Average rank of first non-self match, no duplicates:", np.mean(ranks_list_noduplicates))
print("Median rank of first non-self match:", np.median(ranks_list))
print("Average rank of first non-self match (absolute):", np.mean(first_ranks_list))
print("Median rank of first non-self match (absolute):", np.median(first_ranks_list))

# Compare ranks between w2v and image embeddings
w2v_image = {}
all_sim_w2v = []
all_sim_image = []
for clean_category in word2vec_similarity.index:
    # Get similarity scores for this category only
    word2vec_row = word2vec_similarity.loc[clean_category]
    # Drop NaNs from word2vec
    word2vec_row = word2vec_row.dropna()

    if word2vec_row.empty:
        continue

    word2vec_rank = word2vec_row.rank(ascending=False)
    all_sim_w2v.extend(list(word2vec_row.values))
    word2vec_rank = word2vec_rank[word2vec_rank.index.to_series() != clean_category]

    top5_w2v = word2vec_rank.nsmallest(5)
    top10_w2v = word2vec_rank.nsmallest(10)
    top20_w2v = word2vec_rank.nsmallest(20)

    w2v_image[clean_category] = {}

    pattern = rf"^{re.escape(clean_category)}_"

    image_row = image_similarity.loc[image_similarity.index.str.contains(pattern, regex=True)]

    for i, row in image_row.iterrows():

        # get index name
        cat_row = row.name

        image_row_rank = row.rank(ascending=False)
        
        all_sim_image.extend(list(row.values))
        # Create a DataFrame so we can manipulate it more easily
        image_rank_df = image_row_rank.to_frame(name='rank')
        image_rank_df['column_name'] = image_rank_df.index

        # Extract cluster-less category name
        image_rank_df['category_cleaned'] = image_rank_df['column_name'].str.split("_").str[:-1].str.join("_")

        image_rank_df = image_rank_df[image_rank_df["category_cleaned"] != clean_category]  # remove the category itself

        # Drop duplicates: keep only one entry per category_cleaned (i.e., one cluster per category)
        image_rank_df = image_rank_df.drop_duplicates(subset=['category_cleaned'])

        # Sort by rank (ascending order means highest similarity first)
        image_rank_df.sort_values('rank', inplace=True)

        # create category "random_shuffled_rank" in image_rank_df
        image_rank_df["random_shuffled_rank"] = image_rank_df['rank'].sample(frac=1, random_state=hash(i) % (2**32)).values
        image_rank_df_shuffled = image_rank_df.sort_values('random_shuffled_rank')

        # Get top-K
        top5_image = image_rank_df.head(5)
        top10_image = image_rank_df.head(10)
        top20_image = image_rank_df.head(20)

        top5_image_baseline = image_rank_df_shuffled.head(5)
        top10_image_baseline = image_rank_df_shuffled.head(10)
        top20_image_baseline = image_rank_df_shuffled.head(20)

        rbo_score_5 = rbo.RankingSimilarity(top5_image["category_cleaned"].tolist(), top5_w2v.index.tolist()).rbo()
        rbo_score_10 = rbo.RankingSimilarity(top10_image["category_cleaned"].tolist(), top10_w2v.index.tolist()).rbo()
        rbo_score_20 = rbo.RankingSimilarity(top20_image["category_cleaned"].tolist(), top20_w2v.index.tolist()).rbo()

        rbo_score_5_baseline = rbo.RankingSimilarity(top5_image["category_cleaned"].tolist(), top5_image_baseline["category_cleaned"].tolist()).rbo()
        rbo_score_10_baseline = rbo.RankingSimilarity(top10_image["category_cleaned"].tolist(), top10_image_baseline["category_cleaned"].tolist()).rbo()
        rbo_score_20_baseline = rbo.RankingSimilarity(top20_image["category_cleaned"].tolist(), top20_image_baseline["category_cleaned"].tolist()).rbo()

        overlap_top10 = len(list(set(top10_image["category_cleaned"].tolist()).intersection(set(top10_w2v.index.tolist())))) / len(top10_image)
        overlap_top10_baseline = len(list(set(top10_image["category_cleaned"].tolist()).intersection(list(set(top10_image_baseline["category_cleaned"].tolist()))))) / len(top10_image)

        # Kendall's Tau
        ## Order complete ranks by index
        word2vec_rank = word2vec_rank.sort_index()
        # from image_rank_df, remove categories that are not in word2vec_rank index
        image_rank_df = image_rank_df[image_rank_df['category_cleaned'].isin(word2vec_rank.index)]
        # find out which categories are missing
        missing_categories = set(word2vec_rank.index) - set(image_rank_df['category_cleaned'])
        if missing_categories:
            # remove them from word2vec_rank
            word2vec_rank = word2vec_rank[word2vec_rank.index.isin(image_rank_df['category_cleaned'])]
        image_rank_df = image_rank_df.sort_values(by='category_cleaned')

        kendall_tau = stats.kendalltau(word2vec_rank, image_rank_df['rank'])
        kendall_tau_baseline = stats.kendalltau(image_rank_df['rank'], image_rank_df['random_shuffled_rank'])

        w2v_image[clean_category][cat_row] = {
            "rbo_5": rbo_score_5,
            "rbo_5_baseline": rbo_score_5_baseline,
            "rbo_10": rbo_score_10,
            "rbo_10_baseline": rbo_score_10_baseline,
            "rbo_20": rbo_score_20,
            "rbo_20_baseline": rbo_score_20_baseline,
            "overlap_top10": overlap_top10,
            "overlap_top10_baseline": overlap_top10_baseline,
            "kendall_tau": kendall_tau.correlation,
            "kendall_tau_pvalue": kendall_tau.pvalue,
            "kendall_tau_baseline": kendall_tau_baseline.correlation,
            "kendall_tau_pvalue_baseline": kendall_tau_baseline.pvalue,
        }

# Compare ranks between multilingual and image embeddings
multi_image = {}
all_sim_multi = []
all_sim_image = []
for clean_category in multilingual_similarity.index:
    # Get similarity scores for this category only
    multilingual_row = multilingual_similarity.loc[clean_category]
    # Drop NaNs from multilingual
    multilingual_row = multilingual_row.dropna()

    if multilingual_row.empty:
        continue

    multilingual_rank = multilingual_row.rank(ascending=False)
    all_sim_multi.extend(list(multilingual_row.values))
    multilingual_rank = multilingual_rank[multilingual_rank.index.to_series() != clean_category]

    top5_multi = multilingual_rank.nsmallest(5)
    top10_multi = multilingual_rank.nsmallest(10)
    top20_multi = multilingual_rank.nsmallest(20)

    multi_image[clean_category] = {}

    pattern = rf"^{re.escape(clean_category)}_"

    image_row = image_similarity.loc[image_similarity.index.str.contains(pattern, regex=True)]

    for i, row in image_row.iterrows():

        # get index name
        cat_row = row.name

        image_row_rank = row.rank(ascending=False)
        
        all_sim_image.extend(list(row.values))
        # Create a DataFrame so we can manipulate it more easily
        image_rank_df = image_row_rank.to_frame(name='rank')
        image_rank_df['column_name'] = image_rank_df.index

        # Extract cluster-less category name
        image_rank_df['category_cleaned'] = image_rank_df['column_name'].str.split("_").str[:-1].str.join("_")

        image_rank_df = image_rank_df[image_rank_df["category_cleaned"] != clean_category]  # remove the category itself

        # Drop duplicates: keep only one entry per category_cleaned (i.e., one cluster per category)
        image_rank_df = image_rank_df.drop_duplicates(subset=['category_cleaned'])

        # Sort by rank (ascending order means highest similarity first)
        image_rank_df.sort_values('rank', inplace=True)

        # create category "random_shuffled_rank" in image_rank_df
        image_rank_df["random_shuffled_rank"] = image_rank_df['rank'].sample(frac=1, random_state=hash(i) % (2**32)).values
        image_rank_df_shuffled = image_rank_df.sort_values('random_shuffled_rank')

        # Get top-K
        top5_image = image_rank_df.head(5)
        top10_image = image_rank_df.head(10)
        top20_image = image_rank_df.head(20)

        top5_image_baseline = image_rank_df_shuffled.head(5)
        top10_image_baseline = image_rank_df_shuffled.head(10)
        top20_image_baseline = image_rank_df_shuffled.head(20)

        rbo_score_5 = rbo.RankingSimilarity(top5_image["category_cleaned"].tolist(), top5_multi.index.tolist()).rbo()
        rbo_score_10 = rbo.RankingSimilarity(top10_image["category_cleaned"].tolist(), top10_multi.index.tolist()).rbo()
        rbo_score_20 = rbo.RankingSimilarity(top20_image["category_cleaned"].tolist(), top20_multi.index.tolist()).rbo()

        rbo_score_5_baseline = rbo.RankingSimilarity(top5_image["category_cleaned"].tolist(), top5_image_baseline["category_cleaned"].tolist()).rbo()
        rbo_score_10_baseline = rbo.RankingSimilarity(top10_image["category_cleaned"].tolist(), top10_image_baseline["category_cleaned"].tolist()).rbo()
        rbo_score_20_baseline = rbo.RankingSimilarity(top20_image["category_cleaned"].tolist(), top20_image_baseline["category_cleaned"].tolist()).rbo()

        overlap_top10 = len(list(set(top10_image["category_cleaned"].tolist()).intersection(set(top10_multi.index.tolist())))) / len(top10_image)
        overlap_top10_baseline = len(list(set(top10_image["category_cleaned"].tolist()).intersection(list(set(top10_image_baseline["category_cleaned"].tolist()))))) / len(top10_image)

        # Kendall's Tau
        ## Order complete ranks by index
        multilingual_rank = multilingual_rank.sort_index()
        # from image_rank_df, remove categories that are not in multilingual_rank index
        image_rank_df = image_rank_df[image_rank_df['category_cleaned'].isin(multilingual_rank.index)]
        # find out which categories are missing
        missing_categories = set(multilingual_rank.index) - set(image_rank_df['category_cleaned'])
        if missing_categories:
            # remove them from multilingual_rank
            multilingual_rank = multilingual_rank[multilingual_rank.index.isin(image_rank_df['category_cleaned'])]
        image_rank_df = image_rank_df.sort_values(by='category_cleaned')

        kendall_tau = stats.kendalltau(multilingual_rank, image_rank_df['rank'])
        kendall_tau_baseline = stats.kendalltau(image_rank_df['rank'], image_rank_df['random_shuffled_rank'])

        multi_image[clean_category][cat_row] = {
            "rbo_5": rbo_score_5,
            "rbo_5_baseline": rbo_score_5_baseline,
            "rbo_10": rbo_score_10,
            "rbo_10_baseline": rbo_score_10_baseline,
            "rbo_20": rbo_score_20,
            "rbo_20_baseline": rbo_score_20_baseline,
            "overlap_top10": overlap_top10,
            "overlap_top10_baseline": overlap_top10_baseline,
            "kendall_tau": kendall_tau.correlation,
            "kendall_tau_pvalue": kendall_tau.pvalue,
            "kendall_tau_baseline": kendall_tau_baseline.correlation,
            "kendall_tau_pvalue_baseline": kendall_tau_baseline.pvalue,
        }

sns.histplot(all_sim_multi, bins=20, kde=True, color='skyblue')
plt.xlabel("Value")
plt.ylabel("Density")
plt.title("Histogram with KDE")
plt.xlim((0,1))
plt.savefig("../plots_downstream/distribution_sim_multilingual.pdf")
plt.close()

sns.histplot(all_sim_w2v, bins=20, kde=True, color='skyblue')
plt.xlabel("Value")
plt.ylabel("Density")
plt.title("Histogram with KDE")
plt.xlim((0,1))
plt.savefig("../plots_downstream/distribution_sim_w2v.pdf")
plt.close()

sns.histplot(all_sim_image, bins=20, kde=True, color='skyblue')
plt.xlabel("Value")
plt.ylabel("Density")
plt.title("Histogram with KDE")
plt.xlim((0,1))
plt.savefig("../plots_downstream/distribution_sim_image.pdf")
plt.close()


## w2v vs image
df_w2v_image = create_dataframe(w2v_image)
plot_ranking_comparisons(df_w2v_image, "w2v")

### compute averages per column
df_w2v_image_mean = df_w2v_image[["RBO@5", "RBO@10", "RBO@20", "overlap_top10", "kendall_tau"]].mean()
print(df_w2v_image_mean.mean())

## multi vs image
df_multi_image = create_dataframe(multi_image)
plot_ranking_comparisons(df_multi_image, "multi")

df_multi_image_mean = df_multi_image[["RBO@5", "RBO@10", "RBO@20", "overlap_top10", "kendall_tau"]].mean()
print(df_multi_image_mean.mean())

print((df_w2v_image_mean.mean()+df_multi_image_mean.mean())/2)


