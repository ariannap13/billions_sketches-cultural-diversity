import pandas as pd
import networkx as nx
import numpy as np
from scipy import integrate

use_disparity = True
thresholds = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]  # list of quantile cutoffs
tag = "quantile_alpha"
top_countries = 100

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

for dimension in ["all"]:

    print(f"Processing dimension: {dimension}")

    similarity_matrix = pd.read_csv(
        f"./data/{dimension}/{dimension}_sim_matrix_top{top_countries}.csv", index_col=0
    )

    # clean matrix
    mask = similarity_matrix.copy()
    np.fill_diagonal(mask.values, np.nan)
    valid_countries = mask.notna().any(axis=1)
    matrix_clean = similarity_matrix.loc[valid_countries, valid_countries]

    # build initial graph
    G = nx.Graph()
    for i, c1 in enumerate(matrix_clean.index):
        for j, c2 in enumerate(matrix_clean.columns):
            if i < j:
                weight = matrix_clean.iloc[i, j]
                if weight > 0:
                    G.add_edge(c1, c2, weight=weight)

    if use_disparity:
        G = disparity_filter(G)
        alphas = [d["alpha"] for _, _, d in G.edges(data=True) if "alpha" in d]

        if not alphas:
            print(f"No valid alphas for {dimension}, skipping...")
            continue

        for param in thresholds:
            alpha_thresh = np.quantile(alphas, param)
            print(f"  keeping edges with alpha <= {alpha_thresh:.3f} (quantile {param})")

            # build filtered graph
            G_culture = nx.Graph()
            for u, v, data in G.edges(data=True):
                if data["alpha"] <= alpha_thresh:
                    G_culture.add_edge(u, v, **data)

            # add singleton nodes
            G_culture.add_nodes_from(G.nodes())

            print("top countries:", top_countries, "n. nodes:", G_culture.number_of_nodes())

            # save one GEXF per threshold
            nx.write_gexf(
                G_culture,
                f"../../data/network_top_{top_countries}_countries_culture_top{param}_{tag}_edges.gexf"
            )
