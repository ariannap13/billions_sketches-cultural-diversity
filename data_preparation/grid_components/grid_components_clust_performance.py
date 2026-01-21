import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm

categories = pd.read_csv('../../data/categories_clusterable.csv', names=['category'])

list_scores = []
for category in tqdm(categories['category']):
    for threshold_perc in [60,70,80,90,95]:

        true = pd.read_csv(f'../../data/umap_files/{category}_umap_clusters.csv')
        predicted = pd.read_csv(f'../../data/grid_conn_component/{category}_umap_grid_connected_components_clust_threshold_perc_{threshold_perc}.csv')

        true["assigned"] = [1 if x!= -1 else 0 for x in true['cluster']]
        predicted["assigned"] = [1 if x!= -1 else 0 for x in predicted['connected_component']]

        # Merge true and predicted dataframes on 'key_id', keeping all rows from both
        merge_df = pd.merge(true, predicted, on='key_id', how='outer', suffixes=('_true', '_predicted'))

        merge_df['assigned_true'] = merge_df['assigned_true'].fillna(0)
        merge_df['assigned_predicted'] = merge_df['assigned_predicted'].fillna(0)

        # precision and recall on assigned
        precision = precision_score(merge_df['assigned_true'], merge_df['assigned_predicted'])
        recall = recall_score(merge_df['assigned_true'], merge_df['assigned_predicted'])
        f1 = f1_score(merge_df['assigned_true'], merge_df['assigned_predicted'])

        list_scores.append({
            'category': category,
            'threshold_perc': threshold_perc,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })


df_scores = pd.DataFrame(list_scores)
df_scores.to_csv('../../data/grid_conn_component/precision_recall_f1_clusterable.csv', index=False)