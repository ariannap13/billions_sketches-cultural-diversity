import pandas as pd
import numpy as np
import networkx as nx
from itertools import product
from collections import defaultdict
from scipy import integrate

use_disparity = True
top_countries = 50
dimension = "all"

def disparity_filter(G, weight='weight'):
    ''' Compute significance scores (alpha) for weighted edges in G as defined in Serrano et al. 2009
        Args
            G: Weighted NetworkX graph
        Returns
            Weighted graph with a significance score (alpha) assigned to each edge
        References
            M. A. Serrano et al. (2009) Extracting the Multiscale backbone of complex weighted networks. PNAS, 106:16, pp. 6483-6488.
    '''
    
    if nx.is_directed(G): #directed case    
        N = nx.DiGraph()
        for u in G:
            
            k_out = G.out_degree(u)
            k_in = G.in_degree(u)
            
            if k_out > 1:
                sum_w_out = sum(np.absolute(G[u][v][weight]) for v in G.successors(u))
                for v in G.successors(u):
                    w = G[u][v][weight]
                    p_ij_out = float(np.absolute(w))/sum_w_out
                    alpha_ij_out = 1 - (k_out-1) * integrate.quad(lambda x: (1-x)**(k_out-2), 0, p_ij_out)[0]
                    N.add_edge(u, v, weight = w, alpha_out=float('%.4f' % alpha_ij_out))
                    
            elif k_out == 1 and G.in_degree(G.successors(u)[0]) == 1:
                #we need to keep the connection as it is the only way to maintain the connectivity of the network
                v = G.successors(u)[0]
                w = G[u][v][weight]
                N.add_edge(u, v, weight = w, alpha_out=0., alpha_in=0.)
                #there is no need to do the same for the k_in, since the link is built already from the tail
            
            if k_in > 1:
                sum_w_in = sum(np.absolute(G[v][u][weight]) for v in G.predecessors(u))
                for v in G.predecessors(u):
                    w = G[v][u][weight]
                    p_ij_in = float(np.absolute(w))/sum_w_in
                    alpha_ij_in = 1 - (k_in-1) * integrate.quad(lambda x: (1-x)**(k_in-2), 0, p_ij_in)[0]
                    N.add_edge(v, u, weight = w, alpha_in=float('%.4f' % alpha_ij_in))
        return N
    
    else: #undirected case
        B = nx.Graph()
        for u in G:
            k = len(G[u])
            if k > 1:
                sum_w = sum(np.absolute(G[u][v][weight]) for v in G[u])
                for v in G[u]:
                    w = G[u][v][weight]
                    p_ij = float(np.absolute(w))/sum_w
                    alpha_ij = 1 - (k-1) * integrate.quad(lambda x: (1-x)**(k-2), 0, p_ij)[0]
                    B.add_edge(u, v, weight = w, alpha=float('%.4f' % alpha_ij))
        return B


def make_graph(df, use_disparity=True, top_edges=100):
    """Build network from similarity matrix."""
    G = nx.Graph()
    for i, c1 in enumerate(df.index):
        for j, c2 in enumerate(df.columns):
            if i < j and df.iloc[i, j] > 0:
                G.add_edge(c1, c2, weight=df.iloc[i, j])
    
    if use_disparity:
        G = disparity_filter(G)
        
        # sort edges by alpha (ascending = more significant)
        edges_sorted = sorted(
            G.edges(data=True), key=lambda x: x[2].get("alpha", 1.0)
        )
        
        # keep only top-N edges
        edges_top = edges_sorted[:top_edges]
        
        # build new graph with preserved nodes
        G_top = nx.Graph()
        G_top.add_nodes_from(G.nodes())
        G_top.add_edges_from(edges_top)
    else:
        G_top = G.copy()
    
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

def edge_jaccard(G1, G2):
    E1, E2 = set(G1.edges()), set(G2.edges())
    return len(E1 & E2) / len(E1 | E2) if E1 | E2 else 1.0

def avg_neighborhood_jaccard(G1, G2):
    sims = []
    for node in G1.nodes():
        N1, N2 = set(G1.neighbors(node)), set(G2.neighbors(node))
        sims.append(len(N1 & N2) / len(N1 | N2) if N1 | N2 else 1.0)
    return np.mean(sims)

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


# =============================
# Processing
# =============================

# Image similarities
# Load top countries
top_countries_names = pd.read_csv("../data/country_counts/table_counts.csv")["country"].str.upper().head(top_countries)
# Load similarity data
sim_images = pd.read_csv(f"../data/all_average_sims_top_{top_countries}_countries.csv")
# Compute union of countries in sim_images
countries_in_sims = set(sim_images["country1"]).union(set(sim_images["country2"]))
print("Number of countries in sim_images:", len(countries_in_sims))
# Find top countries missing from sim_images
missing_countries = [c for c in top_countries_names if c not in countries_in_sims]
print("Top countries missing from sim_images:", missing_countries)

sim_images_mat = prepare_similarity_matrix(sim_images)
G_image = make_graph(sim_images_mat, use_disparity, top_edges = 2*top_countries)

# Language similarities
sim_lang = pd.read_csv(f"../data/multilingual_similarity_average_top{top_countries}.csv")
df_country_language = pd.read_csv("../data/country_counts/table_counts.csv")
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
G_language = make_graph(sim_lang_mat, use_disparity, top_edges = 2*top_countries)

# Save networks
for g, name in zip([G_image, G_language], ["image","language"]):
    nx.write_gexf(g, f"../data/{dimension}/network_{name}_top-edges100_topcountries{top_countries}.gexf")

