import logging
import os
import pandas as pd
import re

DATABSE_COLUMNS = [
    "paper ID",
    "paper title",
    "author names",
    "publication venue",
    "year of publication",
]
ORIGINAL_DATABASE_LOCALTIONS = ["data/citation-acm-v8.txt", "data/dblp.txt"]
DATABASES_LOCATIONS = ["data/citation-acm-v8_1995_2004.csv", "data/dblp_1995_2004.csv"]


DEFAULT_ER_CONFIGURATION = {
    "threshold": 0.7,
    "matching_method": "Combined",
    "blocking_method": "FirstLetterTitle",
    "clustering_method": "basic",
    "output_filename": "clustering_results_local.csv",
}

DATA_FOLDER = "data/"
RESULTS_FOLDER = "results/"
FILENAME_LOCAL_MATCHED_ENTITIES = "matched_entities_local.csv"
FILENAME_DP_MATCHED_ENTITIES = "matched_entities_dp.csv"
FILENAME_LOCAL_CLUSTERING = "clustering_results_local.csv"
FILENAME_DP_CLUSTERING = "clustering_results_dp.csv"
FILENAME_ALL_METHODS_RESULTS = "method_results.csv"
FILENAME_SCABILITY_TEST_RESULTS = "scability_results.csv"
FILENAME_DP_LOCAL_DIFFERENCE = "difference_results.csv"


def baseline_filename(matching_method, threshold):
    return f"baseline_{matching_method}_{threshold}.csv"


def matched_entities_filename(blocking_method, matching_method, threshold):
    return f"MatchedEntities_{blocking_method}{matching_method}_{threshold}.csv"


def create_cartesian_product(df1, df2):
    # Create a common key for the Cartesian product
    df1["key"] = 1
    df2["key"] = 1
    # Perform a Cartesian product (cross join) on the common key
    return pd.merge(df1, df2, on="key", suffixes=("_df1", "_df2")).drop("key", axis=1)


# Define a Jaccard similarity function
def jaccard_similarity(set1, set2):
    intersection_size = len(set1.intersection(set2))
    union_size = len(set1.union(set2))

    if union_size == 0:
        return 0.0  # to handle the case when both sets are empty

    return intersection_size / union_size


# Function to find ngrams in a given text
def find_ngrams(text: str, number: int = 3) -> set:
    if not text:
        return set()

    words = [f"  {x} " for x in re.split(r"\W+", text.lower()) if x.strip()]

    ngrams = set()

    for word in words:
        for x in range(0, len(word) - number + 1):
            ngrams.add(word[x : x + number])

    return ngrams


# Function to calculate trigram similarity
def trigram_similarity(text1, text2):
    ngrams1 = find_ngrams(text1)
    ngrams2 = find_ngrams(text2)

    num_unique = len(ngrams1 | ngrams2)
    num_equal = len(ngrams1 & ngrams2)

    return float(num_equal) / float(num_unique)


def test_and_create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        logging.info(f"Folder '{folder_path}' created successfully.")
    else:
        logging.warning(f"Folder '{folder_path}' already exists.")


def save_result(result_df, filename):
    folder_path = RESULTS_FOLDER
    test_and_create_folder(folder_path)
    result_df.to_csv(folder_path + filename, index=False)


def save_data(df, filename):
    folder_path = DATA_FOLDER
    test_and_create_folder(folder_path)
    df.to_csv(folder_path + filename, index=False)


def test():
    logging.info("hello world")


def logging_delimiter(t="=", str="", num=90):
    logging.info(t * (int)((num - len(str)) / 2) + str + t * (int)((num - len(str)) / 2))
