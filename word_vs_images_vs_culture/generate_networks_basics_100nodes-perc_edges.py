import pandas as pd
import numpy as np
import networkx as nx
from itertools import product
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.metrics import normalized_mutual_info_score
import seaborn as sns
import community.community_louvain as community_louvain


top_countries = 100
dimension = "all"

def edges_for_density(density, num_nodes):
    """Compute how many edges correspond to a given density and number of nodes."""
    return int(round(density * num_nodes * (num_nodes - 1) / 2))

def make_graph_baseline(df, top_percent=0.1):
    """
    Build network keeping only the top `top_percent` fraction of edges by weight.
    """
    G = nx.Graph()
    for i, c1 in enumerate(df.index):
        for j, c2 in enumerate(df.columns):
            if i < j and df.iloc[i, j] > 0:
                G.add_edge(c1, c2, weight=df.iloc[i, j])

    # Sort edges by weight descending
    edges_sorted = sorted(G.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)
    n_keep = int(len(edges_sorted) * top_percent)
    edges_to_keep = edges_sorted[:n_keep]

    G_top = nx.Graph()
    G_top.add_nodes_from(G.nodes())
    G_top.add_edges_from([(u, v, d) for u, v, d in edges_to_keep])

    return G_top

def make_graph(df, top_edges=100):
    """Build network from similarity matrix."""
    G = nx.Graph()
    for i, c1 in enumerate(df.index):
        for j, c2 in enumerate(df.columns):
            if i < j and df.iloc[i, j] > 0:
                G.add_edge(c1, c2, weight=df.iloc[i, j])
    
    edges_sorted = sorted(G.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)
    edges_to_keep = edges_sorted[:top_edges]

    G_top = nx.Graph()
    G_top.add_nodes_from(G.nodes())
    G_top.add_edges_from([(u, v, d) for u, v, d in edges_to_keep])

    return G_top

def prepare_similarity_matrix(df, nodes=None, fill_diag=1.0):
    """Pivot long-form similarity table to square matrix, fill lower triangle, set diagonal."""
    mat = df.pivot(index="country1", columns="country2", values=df.columns[-1])
    countries = sorted(set(df["country1"]).union(set(df["country2"])))
    print(len(countries))
    mat = mat.reindex(index=countries, columns=countries)
    mat = mat.combine(mat.T, func=lambda s1, s2: s1.fillna(s2))
    np.fill_diagonal(mat.values, fill_diag)
    if nodes is not None:
        mat = mat.loc[mat.index.isin(nodes), mat.columns.isin(nodes)]
        mat = mat.reindex(index=nodes, columns=nodes)
    return mat

def jaccard_index(set1, set2):
    return len(set1 & set2) / len(set1 | set2) if set1 | set2 else 1.0

def average_best_match_jaccard(comms1, comms2):

    def best_match_avg_weighted(source, target):
        total_weight = sum(len(s) for s in source)
        if total_weight == 0:
            return 0.0
        return sum(len(s) * max(jaccard_index(s, t) for t in target) for s in source) / total_weight

    score_1_to_2 = best_match_avg_weighted(comms1, comms2)
    score_2_to_1 = best_match_avg_weighted(comms2, comms1)
    
    return (score_1_to_2 + score_2_to_1) / 2, (score_1_to_2, score_2_to_1)

def comms_to_labels(comms, nodes):
    node_to_label = {n: i for i, comm in enumerate(comms) for n in comm}
    return [node_to_label.get(n, -1) for n in nodes]

def get_partition_labels(graph):
    """Return node labels from a Louvain community partition, ordered by sorted node IDs."""
    # Compute Louvain partition: dict {node: community_label}
    partition = community_louvain.best_partition(graph, random_state=0)
    # Return labels in sorted node order (important for NMI comparisons)
    return [partition[n] for n in sorted(graph.nodes())]

# =============================
# Processing
# =============================

# Image similarities
# Load top countries
top_countries_names = pd.read_csv("../data/table_counts.csv")["country"].str.upper().head(top_countries)
# Load similarity data
sim_images = pd.read_csv(f"../data/all_average_sims_top_{top_countries}_countries.csv")
# Compute union of countries in sim_images
countries_in_sims = set(sim_images["country1"]).union(set(sim_images["country2"]))
print("Number of countries in sim_images:", len(countries_in_sims))
# Find top countries missing from sim_images
missing_countries = [c for c in top_countries_names if c not in countries_in_sims]
print("Top countries missing from sim_images:", missing_countries)

sim_images_mat = prepare_similarity_matrix(sim_images)

# Language similarities
sim_lang = pd.read_csv(f"../data/multilingual_similarity_average_top{top_countries}.csv")
df_country_language = pd.read_csv("../data/table_counts.csv")
country_to_lang = df_country_language[["country", "final_language"]][:top_countries] \
                .set_index("country")["final_language"] \
                .to_dict()
# Convert keys to uppercase
country_to_lang = {k.upper(): v for k, v in country_to_lang.items()}

language_to_countries = defaultdict(list)
for country, lang in country_to_lang.items():
    language_to_countries[lang].append(country)

# Find all countries appearing in sim_lang
countries_in_sims = set()
for _, row in sim_lang.iterrows():
    countries_in_sims.update(language_to_countries.get(row["lang1"], []))
    countries_in_sims.update(language_to_countries.get(row["lang2"], []))

# Find missing top countries
missing_countries = [c for c in top_countries_names if c not in countries_in_sims]

print("Number of countries in sim_lang:", len(countries_in_sims))
print("Missing top countries:", missing_countries)

# Convert to a normal dictionary
language_to_countries = dict(language_to_countries)
sim_lang.reset_index(inplace=True)

expanded_rows = [
    {"country1": c1, "country2": c2, "average_similarity": row["average_similarity"]}
    for _, row in sim_lang.iterrows()
    for c1, c2 in product(language_to_countries.get(row["lang1"], []), language_to_countries.get(row["lang2"], []))
]

sim_lang_mat = prepare_similarity_matrix(pd.DataFrame(expanded_rows))

diag_results = []
thresholds = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
dict_graphs = {}

for p in thresholds:
    Gp = make_graph_baseline(sim_images_mat, p)
    G_language = make_graph_baseline(sim_lang_mat, p)

    n_nodes = Gp.number_of_nodes()
    n_edges = Gp.number_of_edges()
    isolates = len([n for n, deg in Gp.degree() if deg == 0])
    n_comp = nx.number_connected_components(Gp)
    if n_edges == 0:
        gcc_frac = 0.0
    else:
        gcc = max(nx.connected_components(Gp), key=len, default=set())
        gcc_frac = len(gcc) / n_nodes


    diag_results.append({
        "p": p,
        "n_edges": n_edges,
        "gcc_frac": gcc_frac,
        "n_components": n_comp,
        "n_isolates": isolates,
    })

    dict_graphs[p] = [Gp, G_language]


df_diag = pd.DataFrame(diag_results)

# Compare community similarity across threshold pairs (baseline is 0.1)
labels_base = get_partition_labels(dict_graphs[0.1][0])

nmi_scores = []
for p in thresholds:
    labels_p = get_partition_labels(dict_graphs[p][0])
    nmi = normalized_mutual_info_score(labels_base, labels_p)
    nmi_scores.append((p, nmi))

df_nmi = pd.DataFrame(nmi_scores, columns=["threshold", "nmi"])

plt.figure(figsize=(8, 6))

# plot the NMI values directly
plt.plot(df_nmi["threshold"], df_nmi["nmi"], marker='o')

# axis labels
plt.xlabel('Threshold (% top edges). Anchor is 10% top edges.')
plt.ylabel('NMI')

# add vertical grid lines
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.ylim(0, 1)

sns.despine()
plt.tight_layout()
plt.savefig("../plots_downstream/nmi_image_thresholds.pdf", dpi=300)

# Save networks
for g, name in zip([dict_graphs[0.1][0], dict_graphs[0.1][1]], ["image", "language"]):
    # run Louvain to get partition
    partition = community_louvain.best_partition(g, random_state=0)
    # set node attribute 'community'
    nx.set_node_attributes(g, partition, 'community')
    nx.write_graphml(g, f"../data/{dimension}/network_{name}_top-edges_topcountries100-communities.graphml")
    nx.write_gexf(g, f"../data/{dimension}/network_{name}_top-edges_topcountries100-communities.gexf")

# for param in thresholds:
#     nx.write_gexf(dict_graphs[param][0], f"./data/{dimension}/network_image_top-{param}_edges_topcountries100.gexf")

# # generate language network with the same density
# density_image100 = nx.density(dict_graphs[0.1][0])
# top_edges = edges_for_density(density_image100, top_countries)

# G_language = make_graph(sim_lang_mat, top_edges)
# nx.write_gexf(G_language, f"./data/{dimension}/network_language_top-edges_topcountries100.gexf")