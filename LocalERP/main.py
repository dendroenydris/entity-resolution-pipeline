import pandas as pd
from LocalERP.matching import *
from LocalERP.clustering import clustering_basic, run_clustering

ERconfiguration = {
    "matching_method": "Jaccard",
    "blocking_method": "Year",
    "threshold": 0.5,
    "output_filename": "results/clustering_results.csv",
}


def ER_pipline(dfilename1, dfilename2, ERconfiguration):
    # import database
    df1 = pd.read_csv(dfilename1, sep=",", engine="python")
    df2 = pd.read_csv(dfilename2, sep=",", engine="python")

    df1["index"] = np.arange(len(df1))
    df2["index"] = np.arange(len(df2)) + len(df1)

    similarity_threshold = ERconfiguration["threshold"]
    result_df = blocking(df1, df2, ERconfiguration["blocking_method"])
    result_df = matching(
        result_df, similarity_threshold, ERconfiguration["matching_method"]
    )
    run_clustering(result_df, df1, df2, ERconfiguration['clustering_method'])


if __name__ == "__main__":
    ER_pipline(
        "data/citation-acm-v8_1995_2004.csv", "data/dblp_1995_2004.csv", ERconfiguration
    )


def scability_test():
    from LocalERP.clustering import clustering_basic
    from LocalERP.matching import MATCHING_METHODS,BLOCKING_METHODS

    for method in MATCHING_METHODS:
        for threshold in [0.5, 0.7, 0.8, 0.9]:
            for blocking_method in BLOCKING_METHODS:
                ERconfiguration = {
                    "threshold": threshold,
                    "matching_method": method,
                    "blocking_method": blocking_method,
                    "clustering_method": 'basic',
                    "output_filename": "results/clustering_results.csv",
                }

                ER_pipline(
                    "data/citation-acm-v8_1995_2004.csv",
                    "data/dblp_1995_2004.csv",
                    ERconfiguration,
                )
