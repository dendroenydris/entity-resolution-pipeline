from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, hash, col, substring
from pyspark.sql.types import StringType
from itertools import combinations
from LocalERP.matching import calculate_confusion_matrix
import pandas as pd
from pyspark.sql.window import Window
from graphframes import GraphFrame
from LocalERP.utils import jaccard_similarity, trigram_similarity


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
    return matched_pairs


def create_baseline(df1, df2):
    t_df1 = df1.select([col(col_name).alias(col_name + "_df1") for col_name in df1.columns])
    t_df2 = df2.select([col(col_name).alias(col_name + "_df2") for col_name in df2.columns])

    baseline_pairs = t_df1.crossJoin(t_df2).withColumn(
        "similarity_score", process_udf(col("paper title_df1"), col("paper title_df2"),
                                        col("author names_df1"), col("author names_df2"))
    ).filter(col("similarity_score") > 0.8)

    return baseline_pairs


def calculate_confusion_score(matched_df, baseline_matches):
    matched_df_pd = matched_df.toPandas()
    matched_df_pd["ID"] = matched_df_pd["paper ID_df1"] + matched_df_pd["paper ID_df2"]

    # Write matched pairs to CSV file
    matched_df_pd.to_csv("Matched Entities.csv", index=False)

    baseline_matches_pd = baseline_matches.toPandas()
    baseline_matches_pd["ID"] = (
        baseline_matches_pd["paper ID_df1"] + baseline_matches_pd["paper ID_df2"]
    )
    baseline_matches_pd.to_csv("baseline_matches.csv", index=False)

    baseline_matches_pd = pd.read_csv("baseline_matches.csv", sep=",")
    matched_pairs_pd = pd.read_csv("Matched Entities.csv", sep=",")

    tp, fn, fp, precision, recall, f1 = calculate_confusion_matrix(
        baseline_matches_pd, matched_pairs_pd
    )

    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)


def clustering(df1, df2, matched_df,filename="results/Clustering Results_pyspark.csv"):
    df = df1.union(df2)
    vertices = df.withColumnRenamed("paper ID", "id")
    vertices = vertices.select("id")
    # Create edges DataFrame
    edges = matched_df.select("paper ID_df1", "paper ID_df2")
    edges = edges.withColumnRenamed("paper ID_df1", "src")
    edges = edges.withColumnRenamed("paper ID_df2", "dst")

    # Create a GraphFrame
    graph = GraphFrame(vertices, edges)

    # Find connected components
    connected_components = graph.connectedComponents()
    # print(connected_components.count(), vertices.count(), edges.count())

    # Select the first vertex in every connected component
    first_vertices_df = connected_components.groupBy("component").agg({"id": "min"})

    # Rename the column to "first_vertex"
    first_vertices_df = first_vertices_df.withColumnRenamed("min(id)", "id")

    # Construct the DataFrame
    first_vertices_df = first_vertices_df.orderBy("component")
    first_vertices_df.toPandas().to_csv(
        filename, index="false"
    )


if __name__ == "__main__":
    # Create a SparkSession in local mode
    conf = SparkConf().setAppName("YourAppName").setMaster("local[*]")
    sc = SparkContext(conf=conf)
    sc.setCheckpointDir("inbox")

    spark = SparkSession.builder.appName("Entity Resolution").getOrCreate()
    # Read the datasets from two databases
    df1 = (
        spark.read.option("delimiter", ",")
        .option("header", True)
        .csv("data/citation-acm-v8_1995_2004.csv")
    )
    df2 = (
        spark.read.option("delimiter", ",")
        .option("header", True)
        .csv("data/dblp_1995_2004.csv")
    )

    df1, df2 = FirstLetterBlocking(df1, df2)
    # df1.show(10)
    matched_pairs = FirstLetterMatching(df1, df2, 0.5)
    matched_pairs.show(10)
    baseline_matches = create_baseline(df1, df2)
    calculate_confusion_score(matched_pairs, baseline_matches)
    clustering(df1, df2, matched_pairs)
    spark.stop()
