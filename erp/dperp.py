import logging
from time import time
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, hash, col, substring, monotonically_increasing_id
from pyspark.sql.types import StringType
from erp.matching import calculate_confusion_matrix, resultToString
import pandas as pd
from graphframes import GraphFrame
from erp.utils import (
    FILENAME_DP_MATCHED_ENTITIES,
    jaccard_similarity,
    trigram_similarity,
    DefaultERconfiguration,
)


def calculate_combined_similarity(title1, title2, author_names_df1, author_names_df2):
    title_set1 = set(title1.split())
    title_set2 = set(title2.split())

    jaccard_similarity_title = jaccard_similarity(title_set1, title_set2)

    if author_names_df1 is None or author_names_df2 is None:
        combined_similarity = jaccard_similarity_title
    else:
        trigram_similarity_author = trigram_similarity(
            author_names_df1, author_names_df2
        )
        combined_similarity = (
            0.7 * jaccard_similarity_title + 0.3 * trigram_similarity_author
        )
    return combined_similarity


process_udf = udf(calculate_combined_similarity, StringType())


def FirstLetterBlocking(df1, df2):
    df1 = df1.withColumn("bucket", substring(col("paper title"), 1, 1))
    df2 = df2.withColumn("bucket", substring(col("paper title"), 1, 1))
    return df1, df2


# Assuming similarity_udf is defined elsewhere in your code
def FirstLetterMatching(df1, df2, threshold):
    matched_pairs = None

    for bucket_id in range(26):
        bucket_df1 = df1.filter(df1.bucket == chr(ord("a") + bucket_id))
        bucket_df2 = df2.filter(df2.bucket == chr(ord("a") + bucket_id))

        # Rename columns for df1
        df1_columns = [
            col(col_name).alias(col_name + "_df1") for col_name in bucket_df1.columns
        ]
        bucket_df1 = bucket_df1.select(df1_columns)

        # Rename columns for df2
        df2_columns = [
            col(col_name).alias(col_name + "_df2") for col_name in bucket_df2.columns
        ]
        bucket_df2 = bucket_df2.select(df2_columns)

        # Perform cross join
        pairs = bucket_df1.crossJoin(bucket_df2)
        pairs = pairs.withColumn(
            "similarity_score",
            process_udf(
                col("paper title_df1"),
                col("paper title_df2"),
                col("author names_df1"),
                col("author names_df2"),
            ),
        )

        # Filter based on the similarity score
        pairs = pairs.filter(pairs.similarity_score > threshold)

        # Union the DataFrames
        if matched_pairs == None:
            matched_pairs = pairs
        else:
            matched_pairs = matched_pairs.union(pairs)
    matched_pairs.toPandas().to_csv(FILENAME_DP_MATCHED_ENTITIES, index=False)
    return matched_pairs


def create_baseline(df1, df2):
    t_df1 = df1.select(
        [col(col_name).alias(col_name + "_df1") for col_name in df1.columns]
    )
    t_df2 = df2.select(
        [col(col_name).alias(col_name + "_df2") for col_name in df2.columns]
    )

    baseline_pairs = (
        t_df1.crossJoin(t_df2)
        .withColumn(
            "similarity_score",
            process_udf(
                col("paper title_df1"),
                col("paper title_df2"),
                col("author names_df1"),
                col("author names_df2"),
            ),
        )
        .filter(col("similarity_score") > 0.8)
    )

    return baseline_pairs


def calculate_confusion_score(
    matched_df,
    baseline_matches,
    filename_matched="DP_baseline_matches.csv",
    filename_baseline="DP_baseline_matches.csv",
):
    matched_df_pd = matched_df.toPandas()
    matched_df_pd["ID"] = matched_df_pd["paper ID_df1"] + matched_df_pd["paper ID_df2"]

    # Write matched pairs to CSV file
    matched_df_pd.to_csv("results/" + filename_matched, index=False)

    baseline_matches_pd = baseline_matches.toPandas()
    baseline_matches_pd["ID"] = (
        baseline_matches_pd["paper ID_df1"] + baseline_matches_pd["paper ID_df2"]
    )

    baseline_matches_pd.to_csv("results/" + filename_baseline, index=False)

    baseline_matches_pd = pd.read_csv("results/" + filename_baseline, sep=",")
    matched_pairs_pd = pd.read_csv("results/" + filename_matched, sep=",")

    tp, fn, fp, precision, recall, f1 = calculate_confusion_matrix(
        baseline_matches_pd, matched_pairs_pd
    )

    logging.info("Precision:", precision)
    logging.info("Recall:", recall)
    logging.info("F1-score:", f1)


def clustering(df1, df2, matched_df, filename="clustering Results_DP.csv"):
    df = df1.union(df2)
    vertices = df.withColumnRenamed("index", "id")
    vertices = vertices.select("id")
    # Create edges DataFrame
    edges = matched_df.select("index_df1", "index_df2")
    edges = edges.withColumnRenamed("index_df1", "src")
    edges = edges.withColumnRenamed("index_df2", "dst")

    # Create a GraphFrame
    graph = GraphFrame(vertices, edges)
    # graph = GraphFrame(
    #     vertices, edges.union(edges.selectExpr("dst as src", "src as dst"))
    # )

    # Find connected components
    connected_components = graph.connectedComponents()
    # logging.info(connected_components.count(), vertices.count(), edges.count())

    # Select the first vertex in every connected component
    first_vertices_df = connected_components.groupBy("component").agg({"id": "max"})

    # Rename the column to "first_vertex"
    first_vertices_df = first_vertices_df.withColumnRenamed("max(id)", "id")

    # Construct the DataFrame
    first_vertices_df = first_vertices_df.orderBy("component")
    first_vertices_df.toPandas().to_csv("results/" + filename, index="false")
    logging.info("finish clustering")


def DP_ER_pipline(filename1, filename2, baseline=False, threshold=0.5, cluster=True):
    conf = SparkConf().setAppName("YourAppName").setMaster("local[*]")
    sc = SparkContext(conf=conf)
    sc.setCheckpointDir("inbox")

    spark = SparkSession.builder.appName("Entity Resolution").getOrCreate()
    # Read the datasets from two databases
    start_time = time()
    df1 = spark.read.option("delimiter", ",").option("header", True).csv(filename1)
    df2 = spark.read.option("delimiter", ",").option("header", True).csv(filename2)
    df1 = df1.withColumn("index", monotonically_increasing_id())
    df2 = df2.withColumn("index", monotonically_increasing_id() + df1.count())
    df1, df2 = FirstLetterBlocking(df1, df2)
    matched_pairs = FirstLetterMatching(df1, df2, threshold)
    end_time = time()
    matching_time = end_time - start_time
    if cluster:
        clustering(df1, df2, matched_pairs)
    end_time = time()
    matched_pairs.show(2)

    if baseline:
        baseline_matches = create_baseline(df1, df2)
        return resultToString(
            DefaultERconfiguration,
            -1,
            -1,
            -1,
            baseline_matches,
            matched_df=matched_pairs,
            suffix="_dp",
        )

    out = {
        "dp rate": round(matched_pairs.count() / (df1.count() + df2.count()), 4),
        "dp excution time": round((end_time - start_time) / 60, 2),
        "dp excution time(matching+blocking)": round(matching_time / 60, 2),
    }
    spark.stop()
    return out


if __name__ == "__main__":
    # Create a SparkSession in local mode
    DP_ER_pipline("data/citation-acm-v8_1995_2004.csv", "data/dblp_1995_2004.csv")
