import numpy as np
import pandas as pd
from baseline import calculate_baseline





def run_all_blocking_methods(df1, df2, baseline_configurations, blocking_methods):
    # Create an empty DataFrame to store the results
    results_list = []

    # Iterate through each baseline configuration
    for baseline_config in baseline_configurations:
        # Extract similarity threshold from the current baseline configuration
        similarity_threshold = baseline_config["threshold"]

        # Calculate baseline
        baseline_df, baseline_execution_time = calculate_baseline(
            df1, df2, baseline_config
        )

        # Iterate through each blocking method
        for blocking_method in blocking_methods:
            # Dynamically call the blocking method function
            blocking_function = globals()[f"create_{blocking_method}Blocking"]
            blocking_df, blocking_execution_time = blocking_function(
                df1, df2, similarity_threshold, baseline_config["method"]
            )

            # Calculate confusion matrix
            tp, fn, fp, precision, recall, f1 = calculate_confusion_matrix(
                baseline_df, blocking_df
            )

            # Append results to the list
            results_list.append(
                {
                    "Baseline Method": baseline_config["method"],
                    "Baseline Execution Time": baseline_execution_time,
                    "Similarity Threshold": similarity_threshold,
                    "Blocking Method": blocking_method,
                    "Blocking Execution Time": blocking_execution_time,
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

    # Convert the results list to a DataFrame
    results = pd.DataFrame(results_list)

    # Write results to a single CSV file
    results.to_csv("results/all_blocking_methods_results.csv", index=False)


if __name__ == "__main__":
    # import database
    df1 = pd.read_csv("data/citation-acm-v8_1995_2004.csv", sep=";", engine="python")
    df1["index"] = np.arange(len(df1))
    df2 = pd.read_csv("data/dblp_1995_2004.csv", sep=";", engine="python")
    df2["index"] = np.arange(len(df2)) + len(df1)

    baseline_configurations = [
        {"method": "Jaccard", "threshold": 0.7},
        {"method": "Jaccard", "threshold": 0.5},
        {"method": "Combined", "threshold": 0.5}
        # Add more baseline configurations as needed
    ]

    # Define blocking methods
    blocking_methods = ["Year", "TwoYear", "FirstLetter", "numAuthors"]
    # blocking_methods = ["Year"]

    # Run all blocking methods for each baseline and record results
    run_all_blocking_methods(df1, df2, baseline_configurations, blocking_methods)
