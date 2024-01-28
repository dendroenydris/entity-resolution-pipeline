from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, hash, col
from pyspark.sql.types import StringType
from itertools import combinations
from sklearn.metrics import precision_score, recall_score, f1_score
from matching import calculate_confusion_matrix
import pandas as pd
from graphframes import GraphFrame


# Create a SparkSession in local mode
spark = SparkSession.builder.appName("Entity Resolution").getOrCreate()

# Read the datasets from two databases
df1 = (
    spark.read.option("delimiter", ";")
    .option("header", True)
    .csv("data/citation-acm-v8_1995_2004.csv")
)
df2 = (
    spark.read.option("delimiter", ";")
    .option("header", True)
    .csv("data/dblp_1995_2004.csv")
)

# -----------------------Blocking---------------------------------
# Assign entries to buckets based on blocking keys (e.g., hash-based blocking by year ranges)
df1 = df1.withColumn("bucket", hash(df1["year of publication"]) % 10)
df2 = df2.withColumn("bucket", hash(df2["year of publication"]) % 10)


# -----------------------Matching---------------------------------
# Define similarity function and apply it to all pairs of entities in each bucket
def similarity_function(entity1, entity2):
    # Implement your similarity calculation logic here
    # Compare relevant attributes of the entities and return a similarity score
    # Example: Jaccard similarity
    set1 = set(entity1.split())
    set2 = set(entity2.split())
    return len(set1.intersection(set2)) / len(set1.union(set2))


similarity_udf = udf(similarity_function, StringType())

matched_pairs = []
for bucket_id in range(10):
    bucket_df1 = df1.filter(df1.bucket == bucket_id)
    bucket_df2 = df2.filter(df2.bucket == bucket_id)
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
        similarity_udf(pairs["paper title_df1"], pairs["paper title_df2"]),
    )
    matched_pairs.extend(pairs.filter(pairs.similarity_score > 0.8).collect())

matched_df = spark.createDataFrame(matched_pairs)

# -----------------------baseline---------------------------------
# Calculate match quality metrics:
# Rename columns for df1
df1_columns = [col(col_name).alias(col_name + "_df1") for col_name in df1.columns]
t_df1 = df1.select(df1_columns)

# Rename columns for df2
df2_columns = [col(col_name).alias(col_name + "_df2") for col_name in df2.columns]
t_df2 = df2.select(df2_columns)

baseline_pairs = t_df1.crossJoin(t_df2)
baseline_pairs = baseline_pairs.withColumn(
    "similarity_score",
    similarity_udf(
        baseline_pairs["paper title_df1"], baseline_pairs["paper title_df2"]
    ),
)
baseline_matches = baseline_pairs.filter(baseline_pairs.similarity_score > 0.8)

# ---------------------Matching Results------------------------------
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

# -----------------------Clustering---------------------------------
# Read matched pairs CSV into a PySpark DataFrame
matched_df = spark.read.option("header", True).csv("Matched Entities.csv")
# Create vertices DataFrame
vertices = df1.union(df2)
vertices = vertices.withColumnRenamed("paper ID", "id")
# Create edges DataFrame
edges = matched_df.select("paper ID_df1", "paper ID_df2")
edges = edges.withColumnRenamed("paper ID_df1", "src")
edges = edges.withColumnRenamed("paper ID_df2", "dst")

# Create a GraphFrame
graph = GraphFrame(vertices, edges)

# Find connected components
connected_components = graph.connectedComponents()

# Select the first vertex in every connected component
first_vertices_df = connected_components.groupBy("component").agg({"id": "min"})

# Rename the column to "first_vertex"
first_vertices_df = first_vertices_df.withColumnRenamed("min(id)", "first_vertex")

# Construct the DataFrame
result_df = first_vertices_df.orderBy("component")

# Write the DataFrame to a CSV file
result_df.write.csv("First Vertices.csv", header=True, mode="overwrite")
