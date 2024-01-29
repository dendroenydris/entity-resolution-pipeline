import pandas as pd
import numpy as np
from LocalERP.preparing import DATABSE_COLUMNS


def dfs(graph, node, value, visited, L_propa):
    # Depth-first search function to traverse the graph and update values
    value = L_propa[node] if value < L_propa[node] else value
    visited[node] = True

    for i in range(len(graph)):
        if (graph[node][i] == 1) and (not visited[i]):
            # Traverse the graph recursively
            value = i + 1 if i + 1 > value else value
            value, visited = dfs(graph, i, value, visited, L_propa)
    return value, visited


def propagate(begin_idx, Ladj, L_propa):
    # Function to propagate values through the graph
    L_propa_copy = L_propa.copy()
    visited = [False] * len(Ladj)

    # Depth traversal
    value, visited = dfs(Ladj, begin_idx, begin_idx + 1, visited, L_propa_copy)
    L_propa_copy[visited] = value
    return L_propa_copy


def clustering_basic(result_df, df1, df2):
    # Basic clustering function
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

    # Extract data from the original database
    idx_list = np.unique(L_propa) - 1
    combined_df = pd.concat(
        [df1[df1.index.isin(idx_list)], df2[df2.index.isin(idx_list)]]
    )
    return combined_df


def run_clustering(result_df, df1, df2, clustering_method):
    # Run the clustering function and save the results to a CSV file
    df1["index"] = np.arange(len(df1))
    df2["index"] = np.arange(len(df2)) + len(df1)
    combined_df = clustering_basic(result_df, df1, df2)
    combined_df[DATABSE_COLUMNS].to_csv("results/clustering_results.csv", index=None)
    print("%.2f entities are deleted" % (1 - len(combined_df) / (len(df1) + len(df2))))


if __name__ == "__main__":
    # Read data and run clustering with the basic function
    df1 = pd.read_csv("data/citation-acm-v8_1995_2004.csv", sep=",", engine="python")
    df2 = pd.read_csv("data/dblp_1995_2004.csv", sep=",", engine="python")
    result_df = pd.read_csv("results/MatchedEntities_LetterJaccard.csv")
    run_clustering(result_df, df1, df2, clustering_basic)
