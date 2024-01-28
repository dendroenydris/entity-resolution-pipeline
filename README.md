# DIA project

## User Instuctions

### Installation

```shell
cd path-to-this-project
pip install .
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

## Added by Nevo:


# Entity Resolution Pipelining

We are implementing an Entity Resolution Pipelining on two datasets of citation networks, by ACM and DBLP. Here's an outline of the files and folders structure for the project:

📁 project
  |
  ├─ 📁 LocalERP
  |    |    ├─ 📄 __init__.py
  |    |    ├─ 📄 clustering.py
  |    |    ├─ 📄 main.py
  |    |    ├─ 📄 matching.py
  |    |    ├─ 📄 preparing.py
  |    |    ├─ 📄 testMatching.pt
  |    |    └─ 📄 utils.py
  ├─ 📁 PySpark
  |    |    └─ 📄 DPREP.py
  ├─ 📁 data
  |    |    ├─ 📄 DIA_2023_Exercise.pdf
  |    |    ├─ 📄 citation-acm-v8_1995_2004.csv
  |    |    ├─ 📄 dblp_1995_2004.csv
  |    |    ├─ 📄 test.txt
  |    |    └─ 📄 test_1995_2004.csv
  ├─ 📄 .gitignore
  ├─ 📄 requirements.txt
  ├─ 📄 setup.txt
  ├─ 📄 README.md

## LocalERP

In the LocalERP folder, we have the code for implementing the entity resolution pipelining. We need to first run preparing.py in order to prepare and clean the data. Then, we run main.py with preferred ER configurations. Options are:
- BLOCKING_METHODS = {"Year", "TwoYear", "numAuthors", "FirstLetter"}
- MATCHING_METHODS = {"Jaccard", "Combined"}
- "threshold": number between 0-1

Our choice, based on analyzing different approaches was:
- blocking method: "FirstLetter"
- matching method: "Combined"
- threshold: 0.5

The output will show up in the "results" folder in a file called "clustering_results.csv"

## PySpark

In the PySpark folder, we check and compare our own results with Apache Spark framework, with the file DPREP.py.

## Data

In the data folder, we can find the two datasets we work with, "citation-acm-v8_1995_2004.csv" and "dblp_1995_2004.csv", among other datasets we used which are samples of those datasets, to avoid long computation and complexity. You can also find "DIA_2023_Exercise.pdf" which is the instructions file for this project.

Check requirements.txt before running the code in order to verify compatibility.



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
