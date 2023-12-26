# import pyspark
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col


df1 = pd.read_csv("data/citation-acm-v8_1995_2004.csv", delimiter=";;")
df2 = pd.read_csv("data/dblp_1995_2004.csv", delimiter=";;")
merged_df = pd.concat([df1, df2], ignore_index=True)

spark = SparkSession.builder.appName("AuthorYearBlocking").getOrCreate()

# Assuming merged_df is your Pandas DataFrame
# Convert the Pandas DataFrame to a Spark DataFrame
spark_df = spark.createDataFrame(merged_df)

# Add columns for first and last letters of the author's name
spark_df = spark_df.withColumn("first_letter", col("author names").substr(0, 1))
spark_df = spark_df.withColumn("last_letter", col("author names").substr(-1, 1))

spark_df.select("last_letter").show()

# Partition the DataFrame based on year, first letter, and last letter
partitioned_df = spark_df.repartition("year of publication", "first_letter", "last_letter")

# # Show the resulting DataFrame
partitioned_df.show()

# Stop the Spark session
spark.stop()
