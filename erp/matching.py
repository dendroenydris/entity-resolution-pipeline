import logging
from time import time
import pandas as pd
from erp.utils import (
    FILENAME_ALL_METHODS_RESULTS,
    baseline_filename,
    create_cartesian_product,
    jaccard_similarity,
    matched_entities_filename,
    save_result,
    trigram_similarity,
)

# change here if you would like to asses spesific combnation:

BLOCKING_METHODS = {
    "Year",
    "TwoYear",
    "FirstLetterTitle",
    "LastLetterTitle",
    "FirstOrLastLetterTitle",
    "numAuthors",
    "authorLastName",
    "commonAuthors",
    "commonAndNumAuthors",
}

MATCHING_METHODS = {"Jaccard", "Combined"}


def blocking(df1, df2, blocking_method):
    blocking_function = globals()[f"create_{blocking_method}Blocking"]
    return blocking_function(df1, df2)


def matching(
    blocking_results,
    similarity_threshold,
    matching_method,
    outputfile=None,
):
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
    if outputfile != None:
        save_result(result_df, outputfile)
    # Add a new column 'id' with the addition of 'paper title_df1' and 'paper title_df2'
    return result_df


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


def authorLastName(df):
    last_names_column = []
    for author_names_list in [str(names).split(", ") for names in df["author names"]]:
        if len(author_names_list) == 0:
            last_names_column.append(set())
        else:
            last_names_column.append(
                {name.split(" ")[-1] for name in author_names_list if len(name) > 0}
            )
    return last_names_column


def create_authorLastNameBlocking(df1, df2):
    df1["last name list"] = authorLastName(df1)
    df2["last name list"] = authorLastName(df2)
    result_df = create_cartesian_product(df1, df2)

    def filter_function(l1, l2):
        return (len(l1.intersection(l2)) != 0) or ((len(l1) == 0) and (len(l2) == 0))

    result_df = result_df[
        result_df.apply(
            lambda x: filter_function(x["last name list_df1"], x["last name list_df2"]),
            axis=1,
        )
    ]
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
def create_FirstLetterTitleBlocking(df1, df2):
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


def create_FirstOrLastLetterTitleBlocking(df1, df2):
    result_df = create_cartesian_product(df1, df2)

    # Filter rows based on the starting letter of the paper title
    result_df = result_df[
        (result_df["paper title_df1"].str[0] == result_df["paper title_df2"].str[0])
        | (
            result_df["paper title_df1"].str[-1] == result_df["paper title_df2"].str[-1]
        )
    ]

    # Optionally, reset the index
    result_df = result_df.reset_index(drop=True)

    # Drop rows with NaN values in text columns
    result_df = result_df.dropna(subset=["paper title_df1", "paper title_df2"])
    return result_df


# Function for blocking by last letter of title
def create_LastLetterTitleBlocking(df1, df2):
    result_df = create_cartesian_product(df1, df2)

    # Filter rows based on the starting letter of the paper title
    result_df = result_df[
        result_df["paper title_df1"].str[-1] == result_df["paper title_df2"].str[-1]
    ]

    # Optionally, reset the index
    result_df = result_df.reset_index(drop=True)

    # Drop rows with NaN values in text columns
    result_df = result_df.dropna(subset=["paper title_df1", "paper title_df2"])
    return result_df


def create_commonAndNumAuthorsBlocking(df1, df2):
    result_df = create_cartesian_product(df1, df2)

    # Filter rows based on co-authorship (common authors)
    result_df["common_authors"] = result_df.apply(
        lambda row: set(str(row["author names_df1"]).split(", ")).intersection(
            set(str(row["author names_df2"]).split(", "))
        ),
        axis=1,
    )
    result_df = result_df[result_df["common_authors"].apply(len) > 0]

    # Filter pairs based on the difference in the number of authors
    result_df["num_authors_df1"] = result_df["author names_df1"].apply(
        lambda x: len(str(x).split(", "))
    )
    result_df["num_authors_df2"] = result_df["author names_df2"].apply(
        lambda x: len(str(x).split(", "))
    )
    result_df = result_df[
        abs(result_df["num_authors_df1"] - result_df["num_authors_df2"]) <= 2
    ]

    # Optionally, reset the index
    result_df = result_df.reset_index(drop=True)

    # Drop rows with NaN values in text columns
    result_df = result_df.dropna(subset=["paper title_df1", "paper title_df2"])
    return result_df


def create_commonAuthorsBlocking(df1, df2):
    result_df = create_cartesian_product(df1, df2)
    result_df["common_authors"] = result_df.apply(
        lambda row: set(str(row["author names_df1"]).split(", ")).intersection(
            set(str(row["author names_df2"]).split(", "))
        ),
        axis=1,
    )
    result_df = result_df[result_df["common_authors"].apply(lambda x: len(x) > 0)]
    return result_df


def create_numAuthorsBlocking(df1, df2):
    result_df = create_cartesian_product(df1, df2)

    # Filter pairs based on the difference in the number of authors
    result_df = result_df[
        result_df.apply(
            lambda x: abs(
                len(str(x["author names_df1"]).split(", "))
                - len(str(x["author names_df2"]).split(", "))
            )
            <= 1,
            axis=1,
        )
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


def resultToString(
    ERconfiguration,
    baseline_execution_time,
    blocking_execution_time,
    matching_execution_time,
    baseline_df,
    matched_df,
    suffix="",
):
    tp, fn, fp, precision, recall, f1 = calculate_confusion_matrix(
        baseline_df, matched_df
    )
    return {
        "Blocking method" + suffix: ERconfiguration["blocking_method"],
        "Matching Method" + suffix: ERconfiguration["matching_method"],
        "Baseline Execution Time" + suffix: round(baseline_execution_time / 60, 2),
        "Blocking Execution Time" + suffix: round(blocking_execution_time / 60, 2),
        "Matching Execution Time" + suffix: round(matching_execution_time / 60, 2),
        "Execution Time"
        + suffix: round(
            matching_execution_time / 60 + blocking_execution_time / 60,
            2,
        ),
        "Similarity Threshold" + suffix: ERconfiguration["threshold"],
        "Pairs In Baseline" + suffix: len(baseline_df),
        "Pairs In Blocking" + suffix: len(matched_df),
        "TP" + suffix: tp,
        "FN" + suffix: fn,
        "FP" + suffix: fp,
        "Precision" + suffix: precision,
        "Recall" + suffix: recall,
        "F1 Score" + suffix: f1,
    }


def run_all_blocking_matching_methods(
    df1,
    df2,
    thresholds,
    matching_methods,
    blocking_methods,
    save=True,
    output=FILENAME_ALL_METHODS_RESULTS,
):
    # Create an empty DataFrame to store the results
    results_list = []

    for matching_method in matching_methods:
        for threshold in thresholds:
            start_time = time()
            baseline_df = calculate_baseline(
                df1, df2, {"method": matching_method, "threshold": threshold}
            )
            end_time = time()
            logging.info("finished baseline: " + matching_method)
            save_result(baseline_df, baseline_filename(matching_method, threshold))
            baseline_execution_time = end_time - start_time
            # Iterate through each blocking method
            for blocking_method in blocking_methods:
                # Dynamically call the blocking method function
                start_time = time()
                blocking_df = blocking(df1, df2, blocking_method)
                end_time = time()
                blocking_execution_time = end_time - start_time
                logging.info("finished blocking method: " + blocking_method)

                start_time = time()
                matched_df = matching(blocking_df, threshold, matching_method)
                end_time = time()
                matching_execution_time = end_time - start_time
                logging.info("finished matching method: " + matching_method)
                save_result(
                    matched_df,
                    matched_entities_filename(
                        blocking_method, matching_method, threshold
                    ),
                )
                # Append results to the list
                results_list.append(
                    resultToString(
                        {
                            "blocking_method": blocking_method,
                            "matching_method": matching_method,
                            "threshold": threshold,
                        },
                        baseline_execution_time,
                        blocking_execution_time,
                        matching_execution_time,
                        baseline_df,
                        matched_df,
                    )
                )
                logging.info(results_list[-1])
            # Write results to a single CSV Ffile
    if save:
        save_result(
            pd.DataFrame(results_list),
            output,
        )
    return results_list
