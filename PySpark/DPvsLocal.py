from LocalERP import test
import pandas as pd 
df_dp=pd.read_csv("results/Clustering Results_pyspark.csv")["id"]
df_local=pd.read_csv("results/clustering_results.csv")["paper ID"]

# Convert Series to sets for easy set operations
set_df_dp = set(df_dp)
set_df_local = set(df_local)

# Calculate differences, shared elements, and unique elements
differences = set_df_dp.symmetric_difference(set_df_local)
shared_elements = set_df_dp.intersection(set_df_local)
unique_in_df_dp = set_df_dp.difference(set_df_local)
unique_in_df_local = set_df_local.difference(set_df_dp)

# Print the number of data points in each DataFrame
print(f"Number of data points in df_dp: {len(df_dp)}")
print(f"Number of data points in df_local: {len(df_local)}")


# Print the results
print(f"Number of differences: {len(differences)}")
print(f"Number of shared elements: {len(shared_elements)}")
print(f"Number of elements in df_dp but not in df_local: {len(unique_in_df_dp)}")
print(f"Number of elements in df_local but not in df_dp: {len(unique_in_df_local)}")