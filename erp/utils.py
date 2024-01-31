import os
import pandas as pd
import re

bestF1ERconfiguration = {
    "threshold": 0.7,
    "matching_method": "Combined",
    "blocking_method": "FirstLetter",
    "clustering_method": "basic",
    "output_filename": "results/clustering_results.csv",
}


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
        print(f"Folder '{folder_path}' created successfully.")
    else:
        print(f"Folder '{folder_path}' already exists.")


def save_result(result_df, filename):
    folder_path = "results/"
    test_and_create_folder(folder_path)
    result_df.to_csv(folder_path + filename, index=False)





def save_data(df, filename):
    folder_path = "data/"
    test_and_create_folder(folder_path)
    df.to_csv(folder_path + filename, index=False)


def test():
    print("hello world")