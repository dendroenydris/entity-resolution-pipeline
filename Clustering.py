import pandas as pd
import numpy as np


def dfs(graph, node, value, visited, L_propa):
    value = L_propa[node] if value < L_propa[node] else value
    visited[node] = True

    for i in range(len(graph)):
        if (graph[node][i] == 1) and (not visited[i]):
            # print(node,i)
            value = i + 1 if i + 1 > value else value
            value, visited = dfs(graph, i, value, visited, L_propa)
    return value, visited


def propagate(begin_idx, Ladj, L_propa):
    L_propa_copy = L_propa.copy()
    visited = [False] * len(Ladj)
    # depth traversal
    value, visited = dfs(Ladj, begin_idx, begin_idx + 1, visited, L_propa)
    L_propa[visited] = value
    return L_propa


def clustering_basic(result_df, df1, df2):
    num_nodes = len(df1) + len(df2)
    Ladj = np.zeros(shape=(num_nodes, num_nodes))
    # Construct a graph
    for _, row in result_df.iterrows():
        idx1, idx2 = row["index_df1"], row["index_df2"]
        Ladj[idx1][idx2] = 1
        Ladj[idx2][idx1] = 1

    # Propagate
    L_propa = np.zeros(num_nodes)
    while 0 in L_propa:
        begin_idx = np.argmax(np.where(L_propa == 0, 1, 0))
        L_propa[begin_idx] = begin_idx + 1
        L_propa = propagate(begin_idx, Ladj, L_propa)

    # Extract data from original database
    idx_list = np.unique(L_propa) - 1
    combined_df = pd.concat(
        [df1[df1.index.isin(idx_list)], df2[df2.index.isin(idx_list)]]
    )
    return combined_df


def run_clustering(result_df, df1, df2, clustering_funtion):
    df1["index"] = np.arange(len(df1))
    df2["index"] = np.arange(len(df2)) + len(df1)
    combined_df = clustering_funtion(result_df, df1, df2)
    combined_df.to_csv("results/clustering_results.csv", index=None)

if __name__ == "__main__":
    df1 = pd.read_csv("data/citation-acm-v8_1995_2004.csv", sep=";;", engine="python")
    df2 = pd.read_csv("data/dblp_1995_2004.csv", sep=";;", engine="python")
    result_df = pd.read_csv("results/MatchedEntities_YearJaccard_0.5.csv")
    run_clustering(result_df, df1, df2, clustering_basic)
