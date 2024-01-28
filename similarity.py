import pandas as pd
import re


# Define a Jaccard similarity function
def jaccard_similarity(set1, set2):
    intersection_size = len(set1.intersection(set2))
    union_size = len(set1.union(set2))

    if union_size == 0:
        return 0.0  # to handle the case when both sets are empty

    return intersection_size / union_size


# Define a function to calculate Jaccard similarity
def calculate_jaccard_similarity(row:pd.ArrowDtype):
    # Convert the values to sets and calculate Jaccard similarity
    L = ["paper title"]
    value = 0
    for char in L:
        set1 = set(row[char + "_df1"].split())
        set2 = set(row[char + "_df2"].split())
        value += jaccard_similarity(set1, set2)
    return value / len(L)


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


# Function to calculate combined similarity (modify weights as needed)
def calculate_combined_similarity(row: pd.ArrowDtype):
    title_set1 = set(row["paper title_df1"].split())
    title_set2 = set(row["paper title_df2"].split())
    jaccard_similarity_title = jaccard_similarity(title_set1, title_set2)

    author_names_df1 = row["author names_df1"]
    author_names_df2 = row["author names_df2"]

    if pd.isna(author_names_df1) or pd.isna(author_names_df2):
        combined_similarity = jaccard_similarity_title
    else:
        trigram_similarity_author = trigram_similarity(
            author_names_df1, author_names_df2
        )
        combined_similarity = (
            0.7 * jaccard_similarity_title + 0.3 * trigram_similarity_author
        )

    return combined_similarity
