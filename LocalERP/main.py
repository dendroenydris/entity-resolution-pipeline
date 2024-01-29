from time import time
import pandas as pd
from LocalERP.matching import *
from LocalERP.clustering import clustering_basic, run_clustering
from LocalERP.preparing import prepare_data

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
    run_clustering(result_df, df1, df2, ERconfiguration["clustering_method"])


if __name__ == "__main__":
    ER_pipline(
        "data/citation-acm-v8_1995_2004.csv", "data/dblp_1995_2004.csv", ERconfiguration
    )


def run_all_blocking_matching_methods(
    df1, df2, thresholds, matching_methods, blocking_methods
):
    # Create an empty DataFrame to store the results
    results_list = []
    for threshold in thresholds:
        for matching_method in matching_methods:
            start_time = time()
            baseline_df, baseline_execution_time = calculate_baseline(
                df1, df2, {"method": matching_method, "threshold": threshold}
            )
            end_time = time()
            save_result("results/baseline_{matching_method}_{similarity_threshold}.csv")
            baseline_execution_time = end_time - start_time
            # Iterate through each blocking method
            for blocking_method in blocking_methods:
                similarity_threshold = threshold

                # Dynamically call the blocking method function
                start = time()
                blocking_df = blocking(df1, df2, blocking_method)
                end = time()
                blocking_execution_time = end - start
                start = time()
                matched_df = matching(
                    blocking_df, similarity_threshold, matching_method
                )
                end = time()
                end = time()
                matching_execution_time = end - start
                save_result("MatchedEntities_YearJaccard_{similarity_threshold}.csv")

                # Calculate confusion matrix
                tp, fn, fp, precision, recall, f1 = calculate_confusion_matrix(
                    baseline_df, matched_df
                )

                # Append results to the list
                results_list.append(
                    {
                        "Blocking method": blocking_method,
                        "Blocking Method": matching_method,
                        "Baseline Execution Time": baseline_execution_time,
                        "Blocking Execution Time": blocking_execution_time,
                        "Matching Execution Time": matching_execution_time,
                        "Similarity Threshold": similarity_threshold,
                        "Pairs In Baseline": len(baseline_df),
                        "Pairs In Blocking": len(blocking_df),
                        "TP": tp,
                        "FN": fn,
                        "FP": fp,
                        "Precision": precision,
                        "Recall": recall,
                        "F1 Score": f1,
                    }
                )

    # Write results to a single CSV file
    save_result(
        pd.DataFrame(results_list),
        "results/method_results.csv",
        index=False,
    )


def scability_test():
    from LocalERP.clustering import clustering_basic
    from LocalERP.matching import MATCHING_METHODS, BLOCKING_METHODS

    for method in MATCHING_METHODS:
        for threshold in [0.5, 0.7, 0.8, 0.9]:
            for blocking_method in BLOCKING_METHODS:
                ERconfiguration = {
                    "threshold": threshold,
                    "matching_method": method,
                    "blocking_method": blocking_method,
                    "clustering_method": "basic",
                    "output_filename": "results/clustering_results.csv",
                }

                ER_pipline(
                    "data/citation-acm-v8_1995_2004.csv",
                    "data/dblp_1995_2004.csv",
                    ERconfiguration,
                )


def part1():
    for data in ["data/citation-acm-v8.txt", "data/dblp.txt"]:
        prepare_data(data)

def part2(thresholds=[0.5, 0.7]):
    # import database
    df1 = pd.read_csv("data/citation-acm-v8_1995_2004.csv", sep=",", engine="python")
    df1["index"] = np.arange(len(df1))
    df2 = pd.read_csv("data/dblp_1995_2004.csv", sep=",", engine="python")
    df2["index"] = np.arange(len(df2)) + len(df1)
    # Run all blocking methods for each baseline and record results
    run_all_blocking_matching_methods(
        df1, df2, thresholds, MATCHING_METHODS, BLOCKING_METHODS
    )

