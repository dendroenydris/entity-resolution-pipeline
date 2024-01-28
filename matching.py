import numpy as np
import pandas as pd
from similarity import calculate_combined_similarity, calculate_jaccard_similarity

BLOCKING_METHODS = {"Year", "TwoYear", "numAuthors", "FirstLetter"}
MATCHING_METHODS = {"Jaccard", "Combined"}


def create_cartesian_product(df1, df2):
    # Create a common key for the Cartesian product
    df1["key"] = 1
    df2["key"] = 1
    # Perform a Cartesian product (cross join) on the common key
    return pd.merge(df1, df2, on="key", suffixes=("_df1", "_df2")).drop("key", axis=1)


def matching(blocking_results, similarity_threshold, matching_method):
    result_df = blocking_results.copy()
    # Calculate similarity based on the specified matching method
    if matching_method == "Jaccard":
        result_df["similarity_score"] = result_df.apply(
            calculate_jaccard_similarity, axis=1
        )
    elif matching_method == "Combined":
        result_df["similarity_score"] = result_df.apply(
            calculate_combined_similarity, axis=1
        )

    # Keep rows where similarity is above the threshold
    result_df = result_df[result_df["similarity_score"] > similarity_threshold]

    # Add a new column 'id' with the addition of 'paper title_df1' and 'paper title_df2'
    result_df["ID"] = (
        result_df["paper ID_df1"]
        + result_df["paper ID_df2"]
        + result_df["paper title_df1"]
        + result_df["paper title_df2"]
    )

    file_name = f"MatchedEntities_YearJaccard_{similarity_threshold}.csv"
    # Write matched pairs to a new CSV file
    result_df.to_csv("results/" + file_name, index=False)
    return result_df


def blocking(df1, df2, blocking_method):
    blocking_function = globals()[f"create_{blocking_method}Blocking"]
    return blocking_function(df1, df2)


def create_YearBlocking(df1, df2):
    result_df = create_cartesian_product(df1, df2)

    # Filter rows based on the 'year of publication' column
    result_df = result_df[
        result_df["year of publication_df1"] == result_df["year of publication_df2"]
    ]

    # Optionally, drop duplicate columns and reset the index
    result_df = result_df.drop(["year of publication_df1"], axis=1).reset_index(
        drop=True
    )

    # Rename the columns if needed
    result_df = result_df.rename(
        columns={"year of publication_df2": "year of publication"}
    )

    # Drop rows with NaN values in text columns
    result_df = result_df.dropna(subset=["paper title_df1", "paper title_df2"])
    return result_df


# Function to run a blocking and matching method
def create_TwoYearBlocking(df1, df2, similarity_threshold, matching_method):
    result_df = create_cartesian_product(df1, df2)

    # Filter rows based on the two-year range
    result_df = result_df[
        (
            result_df["year of publication_df2"] - 1
            == result_df["year of publication_df1"]
        )
        | (result_df["year of publication_df2"] == result_df["year of publication_df1"])
    ]

    # Optionally, reset the index
    result_df = result_df.reset_index(drop=True)

    # Drop rows with NaN values in text columns
    result_df = result_df.dropna(subset=["paper title_df1", "paper title_df2"])
    return result_df


# Function for blocking by first letter of title
def create_FirstLetterBlocking(df1, df2, similarity_threshold, matching_method):
    result_df = create_cartesian_product(df1, df2)

    # Filter rows based on the starting letter of the paper title
    result_df = result_df[
        result_df["paper title_df1"].str[0] == result_df["paper title_df2"].str[0]
    ]

    # Optionally, reset the index
    result_df = result_df.reset_index(drop=True)

    # Drop rows with NaN values in text columns
    result_df = result_df.dropna(subset=["paper title_df1", "paper title_df2"])
    return result_df


def create_numAuthorsBlocking(df1, df2, similarity_threshold, matching_method):
    result_df = create_cartesian_product(df1, df2)

    # Filter rows based on co-authorship (common authors)
    result_df["common_authors"] = result_df.apply(
        lambda row: set(row["author names_df1"].split(", ")).intersection(
            set(row["author names_df2"].split(", "))
        ),
        axis=1,
    )
    result_df = result_df[result_df["common_authors"].apply(len) > 0]

    # Filter pairs based on the difference in the number of authors
    result_df["num_authors_df1"] = result_df["author names_df1"].apply(
        lambda x: len(x.split(", "))
    )
    result_df["num_authors_df2"] = result_df["author names_df2"].apply(
        lambda x: len(x.split(", "))
    )
    result_df = result_df[
        abs(result_df["num_authors_df1"] - result_df["num_authors_df2"]) <= 2
    ]

    # Optionally, reset the index
    result_df = result_df.reset_index(drop=True)

    # Drop rows with NaN values in text columns
    result_df = result_df.dropna(subset=["paper title_df1", "paper title_df2"])
    return result_df


def calculate_confusion_matrix(baseline_df, blocked_df):
    inner_join = pd.merge(baseline_df, blocked_df, how="inner", on=["ID"])
    inner_no_duplicates = inner_join.drop_duplicates(subset=["ID"])

    baseline_no_duplicates = baseline_df.drop_duplicates(subset=["ID"])
    tp = inner_no_duplicates.shape[0]

    fn = baseline_no_duplicates.shape[0] - tp

    blocked_no_duplicates = blocked_df.drop_duplicates(subset=["ID"])
    right_join = pd.merge(
        baseline_no_duplicates, blocked_no_duplicates, how="right", on=["ID"]
    )
    fp = right_join.shape[0] - inner_no_duplicates.shape[0]

    # Calculate precision, recall, and F1 score
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (
        (2 * (precision * recall) / (precision + recall))
        if (precision + recall) > 0
        else 0
    )

    return tp, fn, fp, precision, recall, f1


def print_confusion_matrix(
    numbaseline,
    numblocking,
    ERconfiguration,
    baseline_execution_time,
    blocking_execution_time,
    similarity_threshold,
    tp,
    fn,
    fp,
    precision,
    recall,
    f1,
):
    return {
        "Baseline Method": ERconfiguration["baseline_configuration"]["method"],
        "Baseline Execution Time": baseline_execution_time,
        "Similarity Threshold": similarity_threshold,
        "Blocking Method": ERconfiguration["blocking_method"],
        "Blocking Execution Time": blocking_execution_time,
        "Pairs In Baseline": numbaseline,
        "Pairs In Blocking": numblocking,
        "TP": tp,
        "FN": fn,
        "FP": fp,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
    }
