# DIA project

## User Instuctions

### Installation

```shell
cd path-to-this-project
pip install -e .
```

### Sample to Run Exercise

[part1/2](./sample.py)

```python
from LocalERP import part1, part2
part1() # cleaned data stored in "data"
part2() # results of all methods stored in "method_results.csv"
```

for part3

```shell
python -u PySpark/DPREP.py
python -u PySpark/DPvsLocal.py
```

### Quick Introduction

- part1 : `LocalERP.preparing.prepare_data("path_to_txt_file")`
- part2 :
  - Blocking: `LocalERP.blocking(df1,df2,blocking_method)`
    - Parameters:
      - df1,df2 (pandas.DataFrame) : input databases
      - blocking_method(str) : {"Year", "TwoYear", "numAuthors", "FirstLetter"}
  - Matching: `LocalERP.matching(blocking_df,similarity_threshold, matching_method)`
    - Parameters:
      - blocking_df(pandas.DataFrame)
      - similarity_threshold (float from 0.0 to 1.0)
      - matching_method (String) : {"Jaccard", "Combined"}
  - Clustering: `LocalERP.run_clustering(matched_entities, df1, df2, clustering_method)`
    - Parameters:
      - matched_entities(pandas.DataFrame)
      - df1,df2 (pandas.DataFrame) : input databases
      - clustering_method (String) :{'basic'}
  - ER_pipline : `LocalERP.ER_pipline(df1,df2,ERconfiguration)`
    - Parameters:
      - df1,df2 (pandas.DataFrame) : input databases
      - ERconfiguration : sample`{ "matching_method": "Jaccard", "blocking_method": "Year","clustering_method":"basic", "threshold": 0.7, "output_filename": "results/clustering_results.csv",}`
- part 3 :

## Part 1 - Data Acquisition and Preparation !

<p> In this part we obtain the research publication datasets. The datasets are in text format.  
As a prerequisite for <strong>Entity Resolution and model trainin</strong> </p>

We have created a dataset following:

> - paper ID, paper title, author names, publication venue, year of publication
>
> - publications published between 1995 to 2004
>
> - VLDB and SIGMOD venues.

Using Python, we achived resullts of two CSV fills, which would be used as a future datasets.

<img title="" src="https://media.istockphoto.com/id/97980384/photo/mans-hand-squeezing-half-of-lemon.jpg?s=612x612&w=0&k=20&c=fOwBJdxYux4EpCxA5L3zldTuNcJcdKGQuj9JpQTFM6g=" alt="Mans Hand Squeezing Half Of Lemon Stock Photo - Download Image Now - Lemon  - Fruit, Squeezing, Crushed - iStock" width="136" data-align="right">
