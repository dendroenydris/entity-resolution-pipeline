import os
import pandas as pd
import time


# Define a Jaccard similarity function
def jaccard_similarity(set1, set2):
    intersection_size = len(set1.intersection(set2))
    union_size = len(set1.union(set2))

    if union_size == 0:
        return 0.0  # to handle the case when both sets are empty

    return intersection_size / union_size


# Define a function to calculate Jaccard similarity
def calculate_jaccard_similarity(row):
    # Print the values in 'paper title_df1' and 'paper title_df2'
    # print("Paper title_df1:", row['paper title_df1'])
    # print("Paper title_df2:", row['paper title_df2'])

    # Convert the values to sets and calculate Jaccard similarity
    set1 = set(row["paper title_df1"].split())
    set2 = set(row["paper title_df2"].split())
    return jaccard_similarity(set1, set2)


def create_jaccard_baseline(d1, d2):
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
    file_name = f"Jaccard_baseline_{similarity_threshold}.csv"
    # Write matched pairs to a new CSV file
    merged_df.to_csv("results/"+file_name, index=False)
    end_time = time.time()
    execution_time = end_time - start_time
    return merged_df, execution_time


def cosion_similarity():
    return 1


def create_YearComparison(df1, df2, similarity_threshold):
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

    # Calculate Jaccard similarity for each pair in the DataFrame
    result_df["jaccard_similarity"] = result_df.apply(
        calculate_jaccard_similarity, axis=1
    )

    # Keep rows where Jaccard similarity is above the threshold
    result_df = result_df[result_df["jaccard_similarity"] > similarity_threshold]

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


# Function to run a blocking and matching method
def create_TwoYearComparison(df1, df2, similarity_threshold):
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

    # Calculate Jaccard similarity for each pair in the DataFrame
    result_df["jaccard_similarity"] = result_df.apply(
        calculate_jaccard_similarity, axis=1
    )

    # Keep rows where Jaccard similarity is above the threshold
    result_df = result_df[result_df["jaccard_similarity"] > similarity_threshold]

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
# Function for blocking by first letter of title
def create_LetterComparison(df1, df2, similarity_threshold):
    start_time = time.time()

    # Drop rows with NaN values in text columns
    df1 = df1.dropna(subset=['paper title'])
    df2 = df2.dropna(subset=['paper title'])

    # Create a common key for the Cartesian product
    df1['key'] = 1
    df2['key'] = 1

    # Perform a Cartesian product (cross join) on the common key
    result_df = pd.merge(df1, df2, on='key', suffixes=('_df1', '_df2'))

    # Filter rows based on the starting letter of the paper title
    result_df = result_df[result_df['paper title_df1'].str[0] == result_df['paper title_df2'].str[0]]

    # Optionally, reset the index
    result_df = result_df.reset_index(drop=True)

    # Drop rows with NaN values in text columns
    result_df = result_df.dropna(subset=['paper title_df1', 'paper title_df2'])

    # Calculate Jaccard similarity for each pair in the DataFrame
    result_df['jaccard_similarity'] = result_df.apply(calculate_jaccard_similarity, axis=1)

    # Keep rows where Jaccard similarity is above the threshold
    result_df = result_df[result_df['jaccard_similarity'] > similarity_threshold]

    # Add a new column 'id' with the addition of 'paper title_df1' and 'paper title_df2'
    result_df["ID"] = (
            result_df["paper ID_df1"]
            + result_df["paper ID_df2"]
            + result_df["paper title_df1"]
            + result_df["paper title_df2"]
    )

    # Write matched pairs to a new CSV file
    result_df.to_csv("results/"+"MatchedEntities_LetterJaccard.csv", index=False)

    end_time = time.time()
    execution_time = end_time - start_time

    return result_df, execution_time


def create_numAuthorsBlocking(df1, df2):
    start_time = time.time()

    # Drop rows with NaN values in author names columns
    df1 = df1.dropna(subset=['author names'])
    df2 = df2.dropna(subset=['author names'])

    # Create a common key for the Cartesian product
    df1['key'] = 1
    df2['key'] = 1

    # Perform a Cartesian product (cross join) on the common key
    result_df = pd.merge(df1, df2, on='key', suffixes=('_df1', '_df2'))

    # Filter rows based on co-authorship (common authors)
    result_df['common_authors'] = result_df.apply(
        lambda row: set(row['author names_df1'].split(', ')).intersection(set(row['author names_df2'].split(', '))),
        axis=1
    )
    result_df = result_df[result_df['common_authors'].apply(len) > 0]

    # Filter pairs based on the difference in the number of authors
    result_df['num_authors_df1'] = result_df['author names_df1'].apply(lambda x: len(x.split(', ')))
    result_df['num_authors_df2'] = result_df['author names_df2'].apply(lambda x: len(x.split(', ')))
    result_df = result_df[abs(result_df['num_authors_df1'] - result_df['num_authors_df2']) <= 2]

    # Optionally, reset the index
    result_df = result_df.reset_index(drop=True)

    # Drop rows with NaN values in text columns
    result_df = result_df.dropna(subset=['paper title_df1', 'paper title_df2'])

    # Calculate Jaccard similarity for each pair in the DataFrame
    result_df['jaccard_similarity'] = result_df.apply(calculate_jaccard_similarity, axis=1)

    # Keep rows where Jaccard similarity is above the threshold
    result_df = result_df[result_df['jaccard_similarity'] > similarity_threshold]

    # Add a new column 'id' with the addition of 'paper title_df1' and 'paper title_df2'
    result_df["ID"] = (
            result_df["paper ID_df1"]
            + result_df["paper ID_df2"]
            + result_df["paper title_df1"]
            + result_df["paper title_df2"]
    )

    # Write matched pairs to a new CSV file
    result_df.to_csv("results/"+"MatchedEntities_numAuthors.csv", index=False)

    end_time = time.time()
    execution_time = end_time - start_time

    return result_df, execution_time

# def calculate_confusion_matrix(baseline_df, blocked_df, blocking_method):
#     # Check the actual column names in your DataFrames
#
#     # Print the column names to identify the correct common colu
#     # result = pd.merge(left, right, on=["a", "b"])
#     inner_join = pd.merge(baseline_df, blocked_df, how="inner", on=["ID"])
#     inner_no_duplicates = inner_join.drop_duplicates(subset=["ID"])
#
#     baseline_no_duplicates = baseline_df.drop_duplicates(subset=["ID"])
#     tp = inner_no_duplicates.shape[0]
#
#     fn = baseline_no_duplicates.shape[0] - tp
#
#     blocked_no_duplicates = blocked_df.drop_duplicates(subset=["ID"])
#     right_join = pd.merge(
#         baseline_no_duplicates, blocked_no_duplicates, how="right", on=["ID"]
#     )
#     fp = right_join.shape[0] - inner_no_duplicates.shape[0]
#
#     # Set index=False to exclude the index column in the CSV file
#
#     print("tp", tp)
#     print("fn", fn)
#     print("fp", fp)
#
#     # Calculate precision, recall, and F1 score
#     precision_baseline = tp / (tp + fp)
#     recall_baseline = tp / (tp + fn)
#     f1_baseline = (
#         2
#         * (precision_baseline * recall_baseline)
#         / (precision_baseline + recall_baseline)
#         if precision_baseline + recall_baseline > 0
#         else 0
#     )
#
#     # Record results to a CSV file
#     results = pd.DataFrame(
#         {
#             "Blocking Method": [str(blocking_method)],
#             "Matching Method (Baseline)": ["Jaccard Similarity"],
#             "Similarity threshold": [similarity_threshold],
#             "Baseline pairs": [baseline_no_duplicates.shape[0]],
#             "Blocked pairs": [blocked_no_duplicates.shape[0]],
#             "Precision ": [precision_baseline],
#             "Recall ": [recall_baseline],
#             "F1 Score ": [f1_baseline],
#             "Blocking Execution Time ": [blocked_execution_time],
#         }
#     )
#
#     results.to_csv(
#         "results/method_results.csv",
#         mode="a",
#         header=not os.path.exists("method_results.csv"),
#         index=False,
#     )

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
    f1 = (2 * (precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0

    return tp, fn, fp, precision, recall, f1


def run_all_blocking_methods(df1, df2, similarity_threshold):
    baseline_df, baseline_execution_time = create_jaccard_baseline(df1, df2)

    # Run Year Comparison
    year_df, year_execution_time = create_YearComparison(df1, df2, similarity_threshold)
    tp_year, fn_year, fp_year, precision_year, recall_year, f1_year = calculate_confusion_matrix(baseline_df, year_df)

    # Run TwoYear Comparison
    two_year_df, two_year_execution_time = create_TwoYearComparison(df1, df2, similarity_threshold)
    tp_two_year, fn_two_year, fp_two_year, precision_two_year, recall_two_year, f1_two_year = calculate_confusion_matrix(baseline_df, two_year_df)

    # Run Letter Comparison
    letter_df, letter_execution_time = create_LetterComparison(df1, df2, similarity_threshold)
    tp_letter, fn_letter, fp_letter, precision_letter, recall_letter, f1_letter = calculate_confusion_matrix(baseline_df, letter_df)

    # Run Coauthor Comparison
    coauthor_df, coauthor_execution_time = create_numAuthorsBlocking(df1, df2)
    tp_coauthor, fn_coauthor, fp_coauthor, precision_coauthor, recall_coauthor, f1_coauthor = calculate_confusion_matrix(
        baseline_df, coauthor_df)

    # Write results to a single CSV file
    results = pd.DataFrame({
        "Blocking Method": ["Year", "TwoYear", "FirstLetter", "numAuthors"],
        "Similarity Threshold": [similarity_threshold] * 4,
        "Execution Time": [year_execution_time, two_year_execution_time, letter_execution_time,
                           coauthor_execution_time],
        "Pairs Count": [len(year_df), len(two_year_df), len(letter_df), len(coauthor_df)],
        "TP": [tp_year, tp_two_year, tp_letter, tp_coauthor],
        "FN": [fn_year, fn_two_year, fn_letter, fn_coauthor],
        "FP": [fp_year, fp_two_year, fp_letter, fp_coauthor],
        "Precision": [precision_year, precision_two_year, precision_letter, precision_coauthor],
        "Recall": [recall_year, recall_two_year, recall_letter, recall_coauthor],
        "F1 Score": [f1_year, f1_two_year, f1_letter, f1_coauthor],
    })

    results.to_csv("results/all_blocking_methods_results.csv", index=False)

if __name__ == "__main__":
    # import database
    df1 = pd.read_csv("data/citation-acm-v8_1995_2004.csv", sep=";;", engine="python")
    df2 = pd.read_csv("data/dblp_1995_2004.csv", sep=";;", engine="python")
    similarity_threshold = 0.5
    # # Example usage
    # baseline_df, baseline_execution_time = create_jaccard_baseline(df1, df2)
    # # blocked_df, blocked_execution_time = create_YearComparison(df1,df2,similarity_threshold)
    # blocked_df, blocked_execution_time = create_TwoYearComparison(
    #     df1, df2, similarity_threshold
    # )
    # # Run the method and record results
    # calculate_confusion_matrix(baseline_df, blocked_df, "TwoYear")

    # Run all blocking methods and record results
    run_all_blocking_methods(df1, df2, similarity_threshold)
