import numpy as np
import pandas as pd
from time import time
from LocalERP.matching import (
    BLOCKING_METHODS,
    MATCHING_METHODS,
    blocking,
    calculate_baseline,
    calculate_confusion_matrix,
    matching,
)
from LocalERP.utils import save_result


def run_all_blocking_matching_methods(df1, df2, thresholds, matching_methods, blocking_methods):
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
    save_result(pd.DataFrame(results_list),"results/all_blocking_methods_results.csv", index=False)

if __name__ == "__main__":
    # import database
    df1 = pd.read_csv("data/citation-acm-v8_1995_2004.csv", sep=",", engine="python")
    df1["index"] = np.arange(len(df1))
    df2 = pd.read_csv("data/dblp_1995_2004.csv", sep=",", engine="python")
    df2["index"] = np.arange(len(df2)) + len(df1)

    thresholds = [0.5, 0.7]

    # Run all blocking methods for each baseline and record results
    run_all_blocking_matching_methods(df1, df2, thresholds[0], MATCHING_METHODS[0], BLOCKING_METHODS[0])
