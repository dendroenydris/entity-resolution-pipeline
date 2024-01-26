import pandas as pd
from matching import *
from clustering import clustering_basic,run_clustering

ERconfiguration = {
    "baseline_configuration": {"method": "Jaccard", "threshold": 0.5},
    "blocking_method": create_YearComparison,
    "clustering_method": clustering_basic,
    "output_filename": "results/clustering_results.csv",
}


def ER_pipline(dfilename1, dfilename2, ERconfiguration):
    # import database
    df1 = pd.read_csv(dfilename1, sep=";", engine="python")
    df2 = pd.read_csv(dfilename2, sep=";", engine="python")

    df1["index"] = np.arange(len(df1))
    df2["index"] = np.arange(len(df2)) + len(df1)

    similarity_threshold = ERconfiguration["baseline_configuration"]["threshold"]
    result_df, _ = ERconfiguration["blocking_method"](df1, df2, similarity_threshold)

    run_clustering(result_df, df1, df2, ERconfiguration["clustering_method"])


if __name__ == "__main__":
    ER_pipline(
        "data/citation-acm-v8_1995_2004.csv", "data/dblp_1995_2004.csv", ERconfiguration
    )
