import time

import pandas as pd
from similarity import calculate_combined_similarity, calculate_jaccard_similarity


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

def calculate_baseline(df1, df2, baseline_config):
    if baseline_config["method"] == "Jaccard":
        return create_jaccard_baseline(df1, df2, baseline_config["threshold"])
    elif baseline_config["method"] == "Combined":
        return create_combined_baseline(df1, df2, baseline_config["threshold"])
    # Add more baseline methods as needed



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

