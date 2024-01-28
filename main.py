import pandas as pd
from matching import *
from clustering import clustering_basic, run_clustering

ERconfiguration = {
    "baseline_configuration": {"method": "Jaccard", "threshold": 0.5},
    "matching_method": "Jaccard",
    "blocking_method": create_YearBlocking,
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
    result_df, _ = ERconfiguration["blocking_method"](df1, df2, similarity_threshold,ERconfiguration["matching_method"])

    run_clustering(result_df, df1, df2, ERconfiguration["clustering_method"])


if __name__ == "__main__":
    ER_pipline(
        "data/citation-acm-v8_1995_2004.csv", "data/dblp_1995_2004.csv", ERconfiguration
    )


def scability_test():
    from clustering import clustering_basic
    from matching import (
        create_YearBlocking,
        create_FirstLetterBlocking,
        create_numAuthorsBlocking,
        create_TwoYearBlocking,
    )

    for method in ["Jaccard", "Combined"]:
        for threshold in [0.5, 0.7, 0.8, 0.9]:
            for blocking_method in [
                create_YearBlocking,
                create_FirstLetterBlocking,
                create_numAuthorsBlocking,
                create_TwoYearBlocking,
            ]:
                ERconfiguration = {
                    "baseline_configuration": {
                        "method": method,
                        "threshold": threshold,
                    },
                    "matching_method": method,
                    "blocking_method": blocking_method,
                    "clustering_method": clustering_basic,
                    "output_filename": "results/clustering_results.csv",
                }

                ER_pipline(
                    "data/citation-acm-v8_1995_2004.csv",
                    "data/dblp_1995_2004.csv",
                    ERconfiguration,
                )
