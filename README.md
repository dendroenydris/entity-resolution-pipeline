# DIA project

### Quick Overview and User Instruction

<details>
<summary>
Click here to expand!
</summary>

</summary>

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

### Quick Introduction To basic Functions

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

(Can be found [here](https://www.aminer.org/citation))

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

- The steps above will produce the results. They are saved according to your `output_filename` configuration. In our ERconfiguration shown above, it will be saved as `clustering_results.csv` within the `results` folder.

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

We aim to apply an entity resolution pipeline to the aforementioned datasets,
following the scheme depicted below:

![Entity Resolution Pipeline](https://i.ibb.co/bNBH9Xc/Screenshot-2024-01-29-at-15-15-09.png)

_Image Source: Prof. Matthias Boehm, Data Integration and Large-Scale Analysis Course, TU Berlin._

## Prepare Data

Continuing from the previous section, we employ various data cleaning techniques.
It converts all characters to lowercase, ensures uniformity, and eliminates
special characters, retaining only alphanumeric characters, spaces, and commas.
This process standardizes and cleans the textual data for easier comparison
and analysis.

> The code for this part is available in the file named `preparing.py` under
> the function called `prepare_data`.

## Blocking

Blocking is employed to reduce the number of comparisons by using effective
partitioning strategies. In each 'bucket', we run the comparisons
(see section below). Our blocking is achieved through partitioning based on attributes:

1. **Year :** Articles that were published in the same year would be in the same bucket.

2. **Two Year :** Articles that were published in the same year or in the adjacent year would be in the same bucket.

3. **Num Authors :** Articles with a similar number of authors (up to 2 difference) would be in the same bucket.

4. **First Letter :** Articles with the same first letter of title would be in the same bucket.

You can find the code for this part in the file named *`Matching.py.`* Each function is called `*blocking_x,` where x isthe respective blocking method.

## Matching

Before discussing comparison methods, some terms related to our pipeline are
introduced:

- Baseline - We establish a baseline by comparing every pair between datasets, given a certain
similarity function applied. This is our 'ground truth'.
- Prediction - Our model prediction is generated by comparing each pair within a bucket, given the
same similarity function applied to the respective baseline.

**Jaccard -** The Jaccard similarity function is employed to measure the
extent to which two sets share common elements. It does so by calculating
the ratio of the shared elements to the total elements in both sets.
Thresholds of 0.5 and 0.7 are used in the comparison of the 'paper title'
attribute.

**Combined -** This function calculates a combined similarity score
between two papers based on their titles and author names. It utilizes
Jaccard similarity for title comparison and, if available, trigram
similarity for author name comparison. The final combined similarity
score is a weighted sum of title and author name similarities, with
70% weight assigned to the title and 30% to the author names. If author
names are missing for either paper, the function defaults to using
only the Jaccard similarity of titles.

For the blocking methods mentioned above:

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
