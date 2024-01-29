### Entity Resolution of Publication Data

### :books: Table of Contents

* [Abstract](#Abstract)
* [How To Run The Code](#User-Instuctions)
* [Data Acquisition and Preparation (Part 1)](#data-acquisition-and-preparation-part-1)
* [Entity Resolution Pipeline (Part 2)](#entity-resolution-pipeline-part-2)
* [Data Parallel Entity Resolution Pipeline (Part 3)](#data-parallel-entity-resolution-pipeline-part-3)

### :test_tube: Abstract

in this project we were ex

**Motivation:** Our pipeline compares two datasets to group together related papers based on information like paper ID, title, author names, and publication venue. This process ensures we have a clean and accurate dataset, making it easier to study scholarly publications effectively.

### User Instuctions

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

- ? **project**
  - ? **LocalERP**: Contains Python scripts for the entity resolution pipeline.
    - ? `__init__.py`
    - ? `clustering.py`
    - ? `main.py`
    - ? `matching.py`
    - ? `preparing.py`
    - ? `testMatching.py`
    - ? `utils.py`
  - ? **PySpark**: Utilizes Apache Spark for comparison.
    - ? `DPREP.py`
  - ? **data**: Stores datasets and instruction files.
    - ? `DIA_2023_Exercise.pdf`
    - ? `citation-acm-v8_1995_2004.csv`
    - ? `dblp_1995_2004.csv`
    - ? `test.txt`
    - ? `test_1995_2004.csv`
  - ? `.gitignore`
  - ? `requirements.txt`
  - ? `setup.txt`
  - ? `README.md`

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

</details>

</detailes>

---

### Data Acquisition and Preparation (Part 1)

In this part we obtain the research publication datasets. The datasets are in text format. Click here to expand!
As a prerequ?isite for **Entity Resolution and model training**

We have created a dataset following:

> - paper ID, paper title, author names, publication venue, year of publication
> 
> - publications published between 1995 to 2004
> 
> - VLDB and SIGMOD venues.

Using Python and spesficly Pandas DataFrame to covert the datasets from TXT to CSV. Our code iterate the text file with entries separated by double newlines, extracting the attribute above.
It organizes the data into a list of lists, filters based on criteria-1995-2004 publication range and specific venues (SIGMOD or VLDB) and exports the cleaned dataframes to a local folder.



> *You can find the code for this part in the file named `preparing.py`under the function called ``prepare_data.`*
> 
> 
> *Also, CSV files are in your folder with suffix `__1995_2004.csv`*

---

### Entity Resolution Pipeline (Part 2)

We would like to apply an entity resolution pipeline to our two datasets above, following this scheme

<img title="" src="https://i.ibb.co/bNBH9Xc/Screenshot-2024-01-29-at-15-15-09.png" alt="Mans Hand Squeezing Half Of Lemon Stock Photo  Download Image Now  Lemon   Fruit, Squeezing, Crushed  iStock" width="560" data-align="left">

**<u>Prepare Data</u>**

In continuing with the above, the script employs several techniques of **data** **cleaining**. Firstly, it converts all characters to lowercase, ensuring uniformity. 
Special characters are eliminated, leaving only alphanumeric characters, spaces, and commas. This process aids in standardizing and cleaning the textual data, making it easier to compare and analyze. 





> *You can find the code for this part in the file named `preparing.py`under the function is called` prepare_data.*
> 



**<u>Blocking</u>**

We use blocking to reduce the number of comparisons. Instead of comparing every possible pair, we devise effective partitioning strategies. In each partition, we perform the various comparisons (see section below). Our blocking is achieved through partitioning based on attributes:

1. **Year :** Articals that were published in the same year would be in the same bucket.

2. **Two Year :** Articles that were published in the same year or in the adjacent year would be in the same bucket..

3. **Num Authors :** Articals with the same number of autors would be in the same bucket.

4. **First Letter :** Articals with the same first letter would be in the same bucket.
   
   

> *You can find the code for this part in the file named `Matching.py`. 
> Each function is called  `blocking_x`, where x is the respective blocking method.*



**<u>Matching</u>**

Before presenting the comparison methods, let's introduce a few terms related to our piple line:

- Baseline - We establish a baseline by comparing each pair beetwen the datasets.

- Prediction - Our model prediction is generated by comparing each pair within the bucket.

**Jaccard -** We used the famous similarity function - Jaccard, which checks how much two sets share common elements by looking at the ratio of what they have in common to everything they have combined. We used 0.5 and 0.7 thresholds
we conduct with comparing the attrubte 'paper title '.

**Combined -** This function computes a combined similarity score between two papers based on their titles and author names. It employs Jaccard similarity for title comparison and, if available, trigram similarity for author name comparison. The final combined similarity score is a weighted sum of title and author name similarities, with 70% weight given to the title and 30% to the author names. If author names are missing for either paper, the function defaults to using only the Jaccard similarity of titles.



Respectively:
**Jaccard** similarity function with **Year** bucket would yield all The matching articles are those with identical titles and were published in the same year. 

**Jaccard** and **Two year** bucket would yield all The matching articles are those with identical titles and were published in the same year or in the adjacent year
 **Jaccard** and **Num Authors** bucket would yield all The matching articles are those with identical titles and Have the same number of authors.
 **Jaccard** and **First** **Letter** bucket would yeild all The matching articles are those with identical titles and have the same first letter.

As well, the **Combined** would add also the name of the Authors to the above output



> *-You can find the code for this part in the file named Matching.py. 
> Each function is called `calculate_x``, where x is the respective similarity method.*
> 
> *Also, CSV files of each simalrity function and blocking method are in your folder*



Testing each combanition derive these  results :

<img title="" src="https://i.ibb.co/yd5DPGq/Screenshot-2024-01-29-at-15-09-24.png" alt="Mans Hand Squeezing Half Of Lemon Stock Photo  Download Image Now  Lemon   Fruit, Squeezing, Crushed  iStock" width="715">

**<u>Clustering</u>**

and so on

now we would like that

## Data Parallel Entity Resolution Pipeline (Part 3)

kakii
