from scipy.stats import t
import numpy as np
import pandas as pd

def compute_adj_matrix(df, r_thresh=0.8, p_thresh=0.05):
    X = df.values 

    X_std = (X - X.mean(axis=1, keepdims=True)) / X.std(axis=1, ddof=1, keepdims=True)
    R = np.corrcoef(X_std)  
    
    df_t = X.shape[1] - 2 
    with np.errstate(divide='ignore', invalid='ignore'):
        t_stat = R * np.sqrt(df_t / (1 - R**2))
        p_matrix = 2 * t.sf(np.abs(t_stat), df_t)
    
    mask = (p_matrix < p_thresh) & (R > r_thresh)
    
    adj = pd.DataFrame(np.where(mask, R, np.nan),
                       index=df.index, columns=df.index)
    
    np.fill_diagonal(adj.values, np.nan)
    
    return adj



def resolve_collisions(pos, min_dist=0.18, iterations=50):
    nodes = list(pos.keys())
    for _ in range(iterations):
        moved = False
        for i in range(len(nodes)):
            for j in range(i+1, len(nodes)):
                n1, n2 = nodes[i], nodes[j]
                p1, p2 = np.array(pos[n1]), np.array(pos[n2])
                diff = p2 - p1
                dist = np.linalg.norm(diff)

                if dist < min_dist and dist > 0:
                    move = (min_dist - dist) / 2
                    direction = diff / dist
                    pos[n1] = p1 - direction * move
                    pos[n2] = p2 + direction * move
                    moved = True

        if not moved:
            break
    return pos



import networkx as nx
import matplotlib.pyplot as plt

def build_brain_graph(adj_matrix,
                      aggregated_codex,
                      allen_coarse_colors,
                      min_dist=0.5,
                      iterations=60,
                      seed=42,
                      super_scale=5,
                      group_scale=1,
                      node_size=100,
                      name="brain_graph",
                      SAVE=False):

    prv = adj_matrix.fillna(0)

    brain_division = dict(zip(aggregated_codex['target'],
                              aggregated_codex['Coarse']))

    G = nx.from_pandas_adjacency(prv)

    largest_cc = max(nx.connected_components(G), key=len)
    G = G.subgraph(largest_cc).copy()

    divisions = list(set(brain_division.values()))

    supergraph = nx.Graph()
    supergraph.add_nodes_from(divisions)
    supergraph.add_edges_from([
        (divisions[i], divisions[(i + 1) % len(divisions)])
        for i in range(len(divisions))
    ])

    superpos = nx.spring_layout(supergraph, seed=seed, scale=super_scale)

    pos = {}
    for div in divisions:
        group = [n for n in G.nodes() if brain_division[n] == div]
        subG = G.subgraph(group)
        center = superpos[div]
        local_pos = nx.spring_layout(subG, center=center,
                                     scale=group_scale, seed=seed)
        pos.update(local_pos)

    pos = resolve_collisions(pos, min_dist=min_dist, iterations=iterations)

    node_colors = [allen_coarse_colors[brain_division[n]] for n in G.nodes()]

    degrees = dict(G.degree())
    node_sizes = [node_size * np.log1p(degrees[n]) for n in G.nodes()]

    plt.figure(figsize=(10, 10))
    nx.draw_networkx(
        G, pos,
        with_labels=True,
        node_color=node_colors,
        node_size=node_sizes,
        font_family="Arial",
        edge_color="gray",
    )
    plt.axis("off")
    if SAVE:
        plt.savefig(f"{name}.eps")
    plt.show()

    results = {
        "G": G,
        "pos": pos,
        "node_colors": node_colors,
        "divisions": divisions,
        "superpos": superpos,
        "params": {
            "min_dist": min_dist,
            "iterations": iterations,
            "seed": seed,
            "super_scale": super_scale,
            "group_scale": group_scale,
            "node_size": node_size
        }
    }

    return results



def build_brain_graph_no_plot(adj_matrix,
                      aggregated_codex,
                      allen_coarse_colors,
                      min_dist=0.5,
                      iterations=60,
                      seed=42,
                      super_scale=5,
                      group_scale=1,
                      node_size=100):

    prv = adj_matrix.fillna(0)

    brain_division = dict(zip(aggregated_codex['target'],
                              aggregated_codex['Coarse']))

    G = nx.from_pandas_adjacency(prv)

    largest_cc = max(nx.connected_components(G), key=len)
    G = G.subgraph(largest_cc).copy()

    divisions = list(set(brain_division.values()))

    supergraph = nx.Graph()
    supergraph.add_nodes_from(divisions)
    supergraph.add_edges_from([
        (divisions[i], divisions[(i + 1) % len(divisions)])
        for i in range(len(divisions))
    ])

    superpos = nx.spring_layout(supergraph, seed=seed, scale=super_scale)

    pos = {}
    for div in divisions:
        group = [n for n in G.nodes() if brain_division[n] == div]
        subG = G.subgraph(group)
        center = superpos[div]
        local_pos = nx.spring_layout(subG, center=center,
                                     scale=group_scale, seed=seed)
        pos.update(local_pos)

    pos = resolve_collisions(pos, min_dist=min_dist, iterations=iterations)

    node_colors = [allen_coarse_colors[brain_division[n]] for n in G.nodes()]

    results = {
        "G": G,
        "pos": pos,
        "node_colors": node_colors,
        "divisions": divisions,
        "superpos": superpos,
        "params": {
            "min_dist": min_dist,
            "iterations": iterations,
            "seed": seed,
            "super_scale": super_scale,
            "group_scale": group_scale,
            "node_size": node_size
        }
    }

    return results



import community as community_louvain  

def compute_graph_metrics(graph_data):

    G = graph_data["G"]

    metrics = {}

    metrics["density"] = nx.density(G)

    metrics["degree"] = np.mean([d for n, d in G.degree()])

    if nx.is_connected(G):
        metrics["global_efficiency"] = nx.global_efficiency(G)
    else:
        lcc = max(nx.connected_components(G), key=len)
        G_lcc = G.subgraph(lcc)
        metrics["global_efficiency"] = nx.global_efficiency(G_lcc)

    metrics["clustering_coefficient"] = nx.average_clustering(G)

    bc = nx.betweenness_centrality(G)
    metrics["average_betweenness_centrality"] = np.mean(list(bc.values()))
    
    N = G.number_of_nodes()
    k = np.mean([d for n, d in G.degree()])
    C = nx.average_clustering(G)
    if nx.is_connected(G):
        L = nx.average_shortest_path_length(G)
    else:
        lcc_nodes = max(nx.connected_components(G), key=len)
        G_lcc = G.subgraph(lcc_nodes)
        L = nx.average_shortest_path_length(G_lcc)
        
    C_rand = k / N
    L_rand = np.log(N)/np.log(k)
    approx_sigma = (C / C_rand) / (L / L_rand)
    metrics['sigma (small_worldness)'] = approx_sigma
    
    return metrics



from tqdm import tqdm

def bootstrap_adj_matrix(df, r_thresh=0.8, p_thresh=0.05, n_bootstrap=1000, random_state=None):

    rng = np.random.default_rng(random_state)
    adj_list = []

    df = df.apply(pd.to_numeric, errors='coerce')

    for _ in tqdm(range(n_bootstrap), desc="Bootstrapping subjects"):
        sampled_idx = rng.choice(df.columns, size=len(df.columns), replace=True)
        df_boot = df[sampled_idx]

        adj = compute_adj_matrix(df_boot, r_thresh=r_thresh, p_thresh=p_thresh)
        adj_list.append(adj)

    return adj_list



def bootstrap_graph_metrics_subjects(df, aggregated_codex, allen_coarse_colors,
                                     n_bootstrap=1000, random_state=42):

    boot_adj = bootstrap_adj_matrix(df, n_bootstrap=n_bootstrap, random_state=random_state)

    metrics_list = []

    for adj in boot_adj:
        results = build_brain_graph_no_plot(
            adj_matrix=adj,
            aggregated_codex=aggregated_codex,
            allen_coarse_colors=allen_coarse_colors,
            min_dist=0.5,
            iterations=60,
            seed=42,
            super_scale=5,
            group_scale=1,
            node_size=100
        )

        metrics_list.append(compute_graph_metrics(results))

    return pd.DataFrame(metrics_list)


