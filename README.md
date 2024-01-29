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

---

## Quick Project Overview

The project involves implementing an Entity Resolution Pipelining on citation networks from ACM and DBLP datasets.

### Data Source

The project starts with two large dataset text files you need to download:

- [DBLP-Citation-network V8]
- [ACM-Citation-network V8]

(Can be found here - https://www.aminer.org/citation)

Below is the structure of the project:

- 📁 **project**
  - 📁 **LocalERP**: Contains Python scripts for the entity resolution pipeline.
    - 📄 `__init__.py`
    - 📄 `clustering.py`
    - 📄 `main.py`
    - 📄 `matching.py`
    - 📄 `preparing.py`
    - 📄 `testMatching.py`
    - 📄 `utils.py`
  - 📁 **PySpark**: Utilizes Apache Spark for comparison.
    - 📄 `DPREP.py`
  - 📁 **data**: Stores datasets and instruction files.
    - 📄 `DIA_2023_Exercise.pdf`
    - 📄 `citation-acm-v8_1995_2004.csv`
    - 📄 `dblp_1995_2004.csv`
    - 📄 `test.txt`
    - 📄 `test_1995_2004.csv`
  - 📄 `.gitignore`
  - 📄 `requirements.txt`
  - 📄 `setup.txt`
  - 📄 `README.md`

## LocalERP Folder

The LocalERP folder contains scripts for the entity resolution pipeline with specific configurations:

- **Preparing Data**: Run `preparing.prepare_data("path_to_txt_file")` for both text files. This will clean and extract the relevant data (1995-2004 citations by "SIGMOD" or "VLDB" venues). The resulting csv files will show in `data` folder.
- **Running Pipeline**: Execute `main.py` with ER configurations.
- **Configuration Options**:
  - `blocking_method`(String): Methods to reduce execution time {"Year", "TwoYear", "numAuthors", "FirstLetter"}.
  - `matching_method`(String): Algorithms for entity matching {"Jaccard", "Combined"}.
  - `clustering_method`(String): Altogirthm for clustering {"basic"}.
  - `threshold`(float): A value between 0.0-1.0 for the matching similarity threshold.
  - `output_filename`(String): path and file name of results to be saved.

**Selected Configuration**:

ERconfiguration: 
```
{"matching_method": "Combined",
 "blocking_method": "FirstLetter",
 "clustering_method":"basic", 
 "threshold": 0.5, 
 "output_filename": "results/clustering_results.csv"}

```


**Results**:

- The steps above will produce the results. They are saved accordingly to your `output_filename` configuration. In our default case it will be saved as `clustering_results.csv` within the `results` folder.

## PySpark Folder

The PySpark folder contains `DPREP.py` to compare our ER results with the Apache Spark framework.

## Data Folder

The data folder includes the prepared and cleaned datasets and additional samples:

- `citation-acm-v8_1995_2004.csv`: ACM citation network dataset.
- `dblp_1995_2004.csv`: DBLP citation network dataset.
- `DIA_2023_Exercise.pdf`: Project instruction file.


**Note**: Check `requirements.txt` for compatibility before running the code.

---

## DIA Project - ofir's version

### Table of content:

Part 1 - Data Acquisition and Preparation

Part 2 - Entity Resolution Pipeline

Part 3 - Data-Parallel Entity Resolution Pipeline

How To Run The Code

### Part 1 - Data Acquisition and Preparation !

In this part we obtain the research publication datasets. The datasets are in text format. 
As a prerequisite for **Entity Resolution and model trainin**

We have created a dataset following:

> - paper ID, paper title, author names, publication venue, year of publication
> 
> - publications published between 1995 to 2004
> 
> - VLDB and SIGMOD venues.

Using Python, we achived resullts of two CSV fills, which would be used as a future datasets

<img title="" src="https://media.istockphoto.com/id/97980384/photo/mans-hand-squeezing-half-of-lemon.jpg?s=612x612&w=0&k=20&c=fOwBJdxYux4EpCxA5L3zldTuNcJcdKGQuj9JpQTFM6g=" alt="Mans Hand Squeezing Half Of Lemon Stock Photo  Download Image Now  Lemon   Fruit, Squeezing, Crushed  iStock" width="113" data-align="right"> 

### Part 2 - Entity Resolution Pipeline

We would like to apply entity resoltuon pipeline on our 2 data datasets above.
following this scheme:

![](/Users/ofirtopchy/Library/Application%20Support/marktext/images/2024-01-28-22-24-08-image.png)

**Motivation:** Our pipeline compares two datasets to group together related papers based on information like paper ID, title, author names, and publication venue. This process ensures we have a clean and accurate dataset, making it easier to study scholarly publications effectively.

**<u>Prepare Data</u>**

in the part

**<u>Blocking</u>**

We use blocking to reduce the number of comparisons. Instead of comparing every possible pair, we devise effective partitioning strategies. In each partition, we perform the various comparisons (see section below). Our blocking is achieved through partitioning based on attributes:

1. **Year :** Articals that were published in the same year would be in the same bucket.

2. **Two Year :** Articles that were published in the same year or in the adjacent year would be in the same bucket..

3. **Num Authors :** Articals with the same number of autors would be in the same bucket.

4. **First Letter :** Articals with the same first letter would be in the same bucket.

You can find the code for this part in the file named *`Matching.py.`* Each function is called `*blocking_x,` where x isthe respective blocking method.

**<u>Matching</u>**

Before presenting the comparison methods, let's introduce a few terms related to our piple line:

- Baseline - We establish a baseline by comparing each pair beetwen the datasets.

- Prediction - Our model prediction is generated by comparing each pair within the bucket.

**Jaccard -** We used the famous similarity function - Jaccard, which checks how much two sets share common elements by looking at the ratio of what they have in common to everything they have combined. We used 0.5 and 0.7 thresholds
we conduct with comparing the attrubte 'paper title '.

**Combined -** This function computes a combined similarity score between two papers based on their titles and author names. It employs Jaccard similarity for title comparison and, if available, trigram similarity for author name comparison. The final combined similarity score is a weighted sum of title and author name similarities, with 70% weight given to the title and 30% to the author names. If author names are missing for either paper, the function defaults to using only the Jaccard similarity of titles.

If so:

**Jaccard??** similarity function with **Year** bucket would yield all The matching articles are those with identical titles and were published in the same year. **Jaccard??** and **Two year** bucket would yield all The matching articles are those with identical titles and were published in the same year or in the adjacent year **Jaccard??** and **Num Authors** bucket would yield all The matching articles are those with identical titles and Have the same number of authors. **Jaccard** and **First** **Letter** bucket would yeild all The matching articles are those with identical titles and have the same first letter.

As well, the **Combined** would add also the name of the Authors to the above output

*You can find the code for this part in the file named Matching.py.
Each function is called `*?calculate_x``, where x is the respective similarity method.

*You can see the matched entites in CSV file of each simalrity function and blocking method within the reasult folder

Graph

![](/Users/ofirtopchy/Library/Application%20Support/marktext/images/2024-01-28-22-21-15-image.png)

**<u>clustering</u>**

and so on

now we would like that
