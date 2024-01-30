import numpy as np
import pandas as pd
from LocalERP.utils import (
    create_cartesian_product,
    jaccard_similarity,
    save_result,
    trigram_similarity,
)

BLOCKING_METHODS = {"Year", "TwoYear", "numAuthors", "FirstLetter"}
MATCHING_METHODS = {"Jaccard", "Combined"}


# Define a function to calculate Jaccard similarity
def calculate_jaccard_similarity(row: pd.ArrowDtype):
    # Convert the values to sets and calculate Jaccard similarity
    L = ["paper title"]
    value = 0
    for char in L:
        set1 = set(row[char + "_df1"].split())
        set2 = set(row[char + "_df2"].split())
        value += jaccard_similarity(set1, set2)
    return value / len(L)


# Function to calculate combined similarity (modify weights as needed)
def calculate_combined_similarity(row: pd.ArrowDtype):
    title_set1 = set(row["paper title_df1"].split())
    title_set2 = set(row["paper title_df2"].split())
    jaccard_similarity_title = jaccard_similarity(title_set1, title_set2)

    author_names_df1 = row["author names_df1"]
    author_names_df2 = row["author names_df2"]

    if pd.isna(author_names_df1) or pd.isna(author_names_df2):
        combined_similarity = jaccard_similarity_title
    else:
        trigram_similarity_author = trigram_similarity(
            author_names_df1, author_names_df2
        )
        combined_similarity = (
            0.7 * jaccard_similarity_title + 0.3 * trigram_similarity_author
        )

    return combined_similarity


def blocking(df1, df2, blocking_method):
    blocking_function = globals()[f"create_{blocking_method}Blocking"]
    return blocking_function(df1, df2)


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
    return result_df


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
def create_TwoYearBlocking(df1, df2):
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
def create_FirstLetterBlocking(df1, df2):
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


def create_numAuthorsBlocking(df1, df2):
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
    baseline_df["ID"] = baseline_df["paper ID_df1"] + baseline_df["paper ID_df2"]
    blocked_df["ID"] = blocked_df["paper ID_df1"] + blocked_df["paper ID_df2"]
    
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


def calculate_baseline(df1, df2, baseline_config):
    similarity_threshold = baseline_config["threshold"]
    matching_method = baseline_config["method"]
    result_df = create_cartesian_product(df1, df2)
    result_df = matching(result_df, similarity_threshold, matching_method)
    return result_df