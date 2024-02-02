import logging
import pandas as pd
import numpy as np
from erp.utils import DATABSE_COLUMNS, FILENAME_LOCAL_CLUSTERING, save_result

CLUSTERING_METHODS = ["basic"]

def clustering(
    result_df, df1, df2, clustering_method="basic", filename=FILENAME_LOCAL_CLUSTERING
):
    """clustering locally

    Args:
        df1 (DataFrame): database1
        df2 (DataFrame): database2
        matched_df (DataFrame): matched entities
        filename (str, optional):  Defaults to FILENAME_DP_CLUSTERING.
    """
    # Run the clustering function and save the results to a CSV file
    df1["index"] = np.arange(len(df1))
    df2["index"] = np.arange(len(df2)) + len(df1)
    if clustering_method == "basic":
        combined_df = clustering_basic(result_df, df1, df2)
    save_result(combined_df[DATABSE_COLUMNS + ["index"]], filename)
    logging.info(
        "%.2f entities are deleted in clustering." % (1 - len(combined_df) / (len(df1) + len(df2)))
    )
    return combined_df

#==================================================================================
#==============Functions basically never called externally=========================
#==================================================================================

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


def adjs(Ladj, Lsrc):
    Ldst = []
    for i in Lsrc:
        Ldst += [index for index in range(len(Ladj)) if Ladj[i][index] == 1]
    return Ldst


def alladjs(Ladj, src):
    Lsrc = {src}
    Ldst = set(adjs(Ladj, Lsrc))
    Lall = {src}.union(Ldst)
    num = 1
    while len(Lall) != num:
        num = len(Lall)
        Lsrc = Ldst
        Ldst = set(adjs(Ladj, Lsrc))
        Lall.union(Ldst)
    return Lall


def propagate_naive(begin_idx, Ladj, L_propa):
    # Function to propagate values through the graph
    L_propa_copy = L_propa.copy()
    L_adjs = alladjs(Ladj, begin_idx)
    if len(L_adjs) != 1:
        logging.info(L_adjs)
        L_propa_copy[list(L_adjs)] = np.max(list(L_adjs)) + 1
    return L_propa_copy


def test_exist_path(Ladj, src, dest):
    Lsrc = {src}
    Ldst = set(adjs(Ladj, Lsrc))
    Lall = {src}.union(Ldst)
    num = 1
    while len(Lall) != num:
        num = len(Lall)
        if dest in Ldst:
            return True
        Lsrc = Ldst
        Ldst = set(adjs(Ladj, Lsrc))
        Lall.union(Ldst)
    return False


def test_graph(Ladj, L_propa):
    error = 0
    for i in range(len(Ladj)):
        value = L_propa[i] - 1
        if value == i:
            continue
        try:
            assert test_exist_path(Ladj, i, value)
        except AssertionError:
            error += 1
    logging.warning(f"error number {error}")


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

    test_graph(Ladj, L_propa)
    # Extract data from the original database
    idx_list = np.unique(L_propa) - 1
    combined_df = pd.concat(
        [df1[df1.index.isin(idx_list)], df2[df2.index.isin(idx_list)]]
    )
    return combined_df
