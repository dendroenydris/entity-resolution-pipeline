from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, hash, col
from pyspark.sql.types import StringType
from itertools import combinations
from sklearn.metrics import precision_score, recall_score, f1_score
from matching import calculate_confusion_matrix
import pandas as pd

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

# Blocking:
# Assign entries to buckets based on blocking keys (e.g., hash-based blocking by year ranges)
df1 = df1.withColumn("bucket", hash(df1["year of publication"]) % 10)
df2 = df2.withColumn("bucket", hash(df2["year of publication"]) % 10)


# Matching:
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

matched_df = spark.createDataFrame(matched_pairs).toPandas()

matched_df["ID"] = (
    matched_df["paper ID_df1"]
    + matched_df["paper ID_df2"]
    + matched_df["paper title_df1"]
    + matched_df["paper title_df2"]
)

# Write matched pairs to CSV file
matched_df.to_csv("Matched Entities.csv", index=False)

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
baseline_matches = baseline_pairs.filter(
    baseline_pairs.similarity_score > 0.8
).toPandas()
baseline_matches["ID"] = (
    baseline_matches["paper ID_df1"]
    + baseline_matches["paper ID_df2"]
    + baseline_matches["paper title_df1"]
    + baseline_matches["paper title_df2"]
)

baseline_matches.to_csv("baseline_matches.csv", index=False)

# ----------------------------------------------------------------
baseline_matches = pd.read_csv("baseline_matches.csv", sep=",")
matched_pairs = pd.read_csv("Matched Entities.csv", sep=",")
tp, fn, fp, precision, recall, f1 = calculate_confusion_matrix(
    baseline_matches, matched_pairs
)

print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
