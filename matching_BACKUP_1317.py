import os
import numpy as np
import pandas as pd
import time
<<<<<<< HEAD
import re
=======

>>>>>>> 31200adb810810da000b51c52ea1282e5f92b3f6

# Define a Jaccard similarity function
def jaccard_similarity(set1, set2):
    intersection_size = len(set1.intersection(set2))
    union_size = len(set1.union(set2))

    if union_size == 0:
        return 0.0  # to handle the case when both sets are empty

    return intersection_size / union_size


# Define a function to calculate Jaccard similarity
def calculate_jaccard_similarity(row):
    # Convert the values to sets and calculate Jaccard similarity
<<<<<<< HEAD
    L = ["paper title"]
=======
    L = ["paper title", "publication venue"]
>>>>>>> 31200adb810810da000b51c52ea1282e5f92b3f6
    value = 0
    for char in L:
        set1 = set(row[char + "_df1"].split())
        set2 = set(row[char + "_df2"].split())
        value += jaccard_similarity(set1, set2)
    return value / len(L)


def create_jaccard_baseline(df1, df2, similarity_threshold):
    start_time = time.time()
    # Add a common key to perform a cross join product
    df1["key"] = 1
    df2["key"] = 1
    # Perform a Cartesian product (cross join) on the common key
    merged_df = pd.merge(df1, df2, on="key", suffixes=("_df1", "_df2"))
    # Calculate Jaccard similarity for each pair in the DataFrame
    merged_df["jaccard_similarity"] = merged_df.apply(
        calculate_jaccard_similarity, axis=1
    )
    # Keep rows where Jaccard similarity is above the threshold
    merged_df = merged_df[merged_df["jaccard_similarity"] > similarity_threshold]
    # Drop the common key
    merged_df.drop("key", axis=1, inplace=True)

    # Add a new column 'id' with the addition of 'paper title_df1' and 'paper title_df2'
    merged_df["ID"] = (
        merged_df["paper ID_df1"]
        + merged_df["paper ID_df2"]
        + merged_df["paper title_df1"]
        + merged_df["paper title_df2"]
    )

    # Construct the new file name with the variable
    file_name = f"results/Jaccard_baseline_{similarity_threshold}.csv"
    # Write matched pairs to a new CSV file
    merged_df.to_csv(file_name, index=False)
    end_time = time.time()
    execution_time = end_time - start_time
    return merged_df, execution_time

<<<<<<< HEAD
# Function to find ngrams in a given text
def find_ngrams(text: str, number: int=3) -> set:
    if not text:
        return set()

    words = [f'  {x} ' for x in re.split(r'\W+', text.lower()) if x.strip()]

    ngrams = set()

    for word in words:
        for x in range(0, len(word) - number + 1):
            ngrams.add(word[x:x+number])

    return ngrams

# Function to calculate trigram similarity
def trigram_similarity(text1, text2):
    ngrams1 = find_ngrams(text1)
    ngrams2 = find_ngrams(text2)

    num_unique = len(ngrams1 | ngrams2)
    num_equal = len(ngrams1 & ngrams2)

    return float(num_equal) / float(num_unique)

# Function to calculate combined similarity (modify weights as needed)
def calculate_combined_similarity(row):
    title_set1 = set(row["paper title_df1"].split())
    title_set2 = set(row["paper title_df2"].split())
    jaccard_similarity_title = jaccard_similarity(title_set1, title_set2)

    author_names_df1 = row["author names_df1"]
    author_names_df2 = row["author names_df2"]

    if pd.isna(author_names_df1) or pd.isna(author_names_df2):
        combined_similarity = jaccard_similarity_title
    else:
        trigram_similarity_author = trigram_similarity(author_names_df1, author_names_df2)
        combined_similarity = 0.7 * jaccard_similarity_title + 0.3 * trigram_similarity_author

    return combined_similarity

#
def create_combined_baseline(df1, df2, similarity_threshold):
    start_time = time.time()
    # Add a common key to perform a cross join product
    df1["key"] = 1
    df2["key"] = 1
    # Perform a Cartesian product (cross join) on the common key
    merged_df = pd.merge(df1, df2, on="key", suffixes=("_df1", "_df2"))
    # Calculate combined similarity for each pair in the DataFrame
    merged_df["combined_similarity"] = merged_df.apply(
        calculate_combined_similarity, axis=1
    )
    # Keep rows where combined similarity is above the threshold
    merged_df = merged_df[merged_df["combined_similarity"] > similarity_threshold]
    # Drop the common key
    merged_df.drop("key", axis=1, inplace=True)

    # Add a new column 'id' with the addition of 'paper title_df1' and 'paper title_df2'
    merged_df["ID"] = (
        merged_df["paper ID_df1"]
        + merged_df["paper ID_df2"]
        + merged_df["paper title_df1"]
        + merged_df["paper title_df2"]
    )

    # Construct the new file name with the variable
    file_name = f"results/Combined_baseline_{similarity_threshold}.csv"
    # Write matched pairs to a new CSV file
    merged_df.to_csv(file_name, index=False)
    end_time = time.time()
    execution_time = end_time - start_time
    return merged_df, execution_time


def create_YearComparison(df1, df2, similarity_threshold, matching_method):
=======

def create_cosine_baseline(df1, df2, similarity_threshold):
    return 1, 2  # Should return the merged_df and execution time, like in jaccard


def create_YearComparison(df1, df2, similarity_threshold):
>>>>>>> 31200adb810810da000b51c52ea1282e5f92b3f6
    start_time = time.time()

    # Create a common key for the Cartesian product
    df1["key"] = 1
    df2["key"] = 1

    # Perform a Cartesian product (cross join) on the common key
    result_df = pd.merge(df1, df2, on="key", suffixes=("_df1", "_df2"))

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

<<<<<<< HEAD
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
=======
    # Calculate Jaccard similarity for each pair in the DataFrame
    result_df["jaccard_similarity"] = result_df.apply(
        calculate_jaccard_similarity, axis=1
    )

    # Keep rows where Jaccard similarity is above the threshold
    result_df = result_df[result_df["jaccard_similarity"] > similarity_threshold]
>>>>>>> 31200adb810810da000b51c52ea1282e5f92b3f6

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

    end_time = time.time()
    execution_time = end_time - start_time

    return result_df, execution_time

<<<<<<< HEAD
# Function to run a blocking and matching method
def create_TwoYearComparison(df1, df2, similarity_threshold, matching_method):
=======

# Function to run a blocking and matching method
def create_TwoYearComparison(df1, df2, similarity_threshold):
>>>>>>> 31200adb810810da000b51c52ea1282e5f92b3f6
    start_time = time.time()

    # Create a common key for the Cartesian product
    df1["key"] = 1
    df2["key"] = 1

    # Perform a Cartesian product (cross join) on the common key
    result_df = pd.merge(df1, df2, on="key", suffixes=("_df1", "_df2"))

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

<<<<<<< HEAD
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
=======
    # Calculate Jaccard similarity for each pair in the DataFrame
    result_df["jaccard_similarity"] = result_df.apply(
        calculate_jaccard_similarity, axis=1
    )

    # Keep rows where Jaccard similarity is above the threshold
    result_df = result_df[result_df["jaccard_similarity"] > similarity_threshold]
>>>>>>> 31200adb810810da000b51c52ea1282e5f92b3f6

    # Add a new column 'id' with the addition of 'paper title_df1' and 'paper title_df2'
    result_df["ID"] = (
        result_df["paper ID_df1"]
        + result_df["paper ID_df2"]
        + result_df["paper title_df1"]
        + result_df["paper title_df2"]
    )

    # Write matched pairs to a new CSV file
    file_name = f"MatchedEntities_TwoYearJaccard_{similarity_threshold}.csv"
    # Write matched pairs to a new CSV file
    result_df.to_csv("results/" + file_name, index=False)

    end_time = time.time()
    execution_time = end_time - start_time

    return result_df, execution_time

<<<<<<< HEAD
# Function for blocking by first letter of title
def create_FirstLetterComparison(df1, df2, similarity_threshold, matching_method):
=======

# Function for blocking by first letter of title
def create_FirstLetterComparison(df1, df2, similarity_threshold):
>>>>>>> 31200adb810810da000b51c52ea1282e5f92b3f6
    start_time = time.time()

    # Drop rows with NaN values in text columns
    df1 = df1.dropna(subset=["paper title"])
    df2 = df2.dropna(subset=["paper title"])

    # Create a common key for the Cartesian product
    df1["key"] = 1
    df2["key"] = 1

    # Perform a Cartesian product (cross join) on the common key
    result_df = pd.merge(df1, df2, on="key", suffixes=("_df1", "_df2"))

    # Filter rows based on the starting letter of the paper title
    result_df = result_df[
        result_df["paper title_df1"].str[0] == result_df["paper title_df2"].str[0]
    ]

    # Optionally, reset the index
    result_df = result_df.reset_index(drop=True)

    # Drop rows with NaN values in text columns
    result_df = result_df.dropna(subset=["paper title_df1", "paper title_df2"])

<<<<<<< HEAD
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
=======
    # Calculate Jaccard similarity for each pair in the DataFrame
    result_df["jaccard_similarity"] = result_df.apply(
        calculate_jaccard_similarity, axis=1
    )

    # Keep rows where Jaccard similarity is above the threshold
    result_df = result_df[result_df["jaccard_similarity"] > similarity_threshold]
>>>>>>> 31200adb810810da000b51c52ea1282e5f92b3f6

    # Add a new column 'id' with the addition of 'paper title_df1' and 'paper title_df2'
    result_df["ID"] = (
        result_df["paper ID_df1"]
        + result_df["paper ID_df2"]
        + result_df["paper title_df1"]
        + result_df["paper title_df2"]
    )

    # Write matched pairs to a new CSV file
    result_df.to_csv("MatchedEntities_LetterJaccard.csv", index=False)

    end_time = time.time()
    execution_time = end_time - start_time

    return result_df, execution_time


<<<<<<< HEAD
def create_numAuthorsComparison(df1, df2, similarity_threshold, matching_method):
=======
def create_numAuthorsComparison(df1, df2, similarity_threshold):
>>>>>>> 31200adb810810da000b51c52ea1282e5f92b3f6
    start_time = time.time()

    # Drop rows with NaN values in author names columns
    df1 = df1.dropna(subset=["author names"])
    df2 = df2.dropna(subset=["author names"])

    # Create a common key for the Cartesian product
    df1["key"] = 1
    df2["key"] = 1

    # Perform a Cartesian product (cross join) on the common key
    result_df = pd.merge(df1, df2, on="key", suffixes=("_df1", "_df2"))

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

<<<<<<< HEAD
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
=======
    # Calculate Jaccard similarity for each pair in the DataFrame
    result_df["jaccard_similarity"] = result_df.apply(
        calculate_jaccard_similarity, axis=1
    )

    # Keep rows where Jaccard similarity is above the threshold
    result_df = result_df[result_df["jaccard_similarity"] > similarity_threshold]
>>>>>>> 31200adb810810da000b51c52ea1282e5f92b3f6

    # Add a new column 'id' with the addition of 'paper title_df1' and 'paper title_df2'
    result_df["ID"] = (
        result_df["paper ID_df1"]
        + result_df["paper ID_df2"]
        + result_df["paper title_df1"]
        + result_df["paper title_df2"]
    )

    # Write matched pairs to a new CSV file
    result_df.to_csv("MatchedEntities_numAuthors.csv", index=False)

    end_time = time.time()
    execution_time = end_time - start_time

    return result_df, execution_time


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
            blocking_function = globals()[f"create_{blocking_method}Comparison"]
            blocking_df, blocking_execution_time = blocking_function(
<<<<<<< HEAD
                df1, df2, similarity_threshold, baseline_config["method"]
=======
                df1, df2, similarity_threshold
>>>>>>> 31200adb810810da000b51c52ea1282e5f92b3f6
            )

            # Calculate confusion matrix
            tp, fn, fp, precision, recall, f1 = calculate_confusion_matrix(
                baseline_df, blocking_df
            )

            # Append results to the list
            results_list.append(
                {
                    "Baseline Method": baseline_config["method"],
<<<<<<< HEAD
                    "Baseline Execution Time": baseline_execution_time,
                    "Similarity Threshold": similarity_threshold,
                    "Blocking Method": blocking_method,
                    "Blocking Execution Time": blocking_execution_time,
                    "Pairs In Baseline": len(baseline_df),
                    "Pairs In Blocking": len(blocking_df),
=======
                    "Similarity Threshold": similarity_threshold,
                    "Blocking Method": blocking_method,
                    "Execution Time": blocking_execution_time,
                    "Pairs Count": len(blocking_df),
>>>>>>> 31200adb810810da000b51c52ea1282e5f92b3f6
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


def calculate_baseline(df1, df2, baseline_config):
    if baseline_config["method"] == "Jaccard":
        return create_jaccard_baseline(df1, df2, baseline_config["threshold"])
<<<<<<< HEAD
    elif baseline_config["method"] == "Combined":
        return create_combined_baseline(df1, df2, baseline_config["threshold"])
=======
    elif baseline_config["method"] == "Cosine":
        return create_cosine_baseline(df1, df2, baseline_config["threshold"])
>>>>>>> 31200adb810810da000b51c52ea1282e5f92b3f6
    # Add more baseline methods as needed


if __name__ == "__main__":
    # import database
    df1 = pd.read_csv("data/citation-acm-v8_1995_2004.csv", sep=";", engine="python")
    df1["index"] = np.arange(len(df1))
    df2 = pd.read_csv("data/dblp_1995_2004.csv", sep=";", engine="python")
    df2["index"] = np.arange(len(df2)) + len(df1)

    baseline_configurations = [
        {"method": "Jaccard", "threshold": 0.7},
        {"method": "Jaccard", "threshold": 0.5},
<<<<<<< HEAD
        {"method": "Combined", "threshold": 0.5}
=======
        # {"method": "Cosine", "threshold": 0.5},  # comment to be removed when Cosine baseline is written.
>>>>>>> 31200adb810810da000b51c52ea1282e5f92b3f6
        # Add more baseline configurations as needed
    ]

    # Define blocking methods
<<<<<<< HEAD
    blocking_methods = ["Year", "TwoYear", "FirstLetter", "numAuthors"]
    # blocking_methods = ["Year"]
=======
    # blocking_methods = ["Year", "TwoYear", "FirstLetter", "numAuthors"]
    blocking_methods = ["Year"]
>>>>>>> 31200adb810810da000b51c52ea1282e5f92b3f6

    # Run all blocking methods for each baseline and record results
    run_all_blocking_methods(df1, df2, baseline_configurations, blocking_methods)
