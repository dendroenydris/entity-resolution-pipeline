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


## Project Structure Overview
The project involves implementing an Entity Resolution Pipelining on citation networks from ACM and DBLP datasets.

### Data Source
The project starts with two large dataset files:
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
    - 📄 `testMatching.pt`
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

- **Preparing Data**: Run `preparing.py` to clean and extract the relevant data (1995-2004 citations by "SIGMOD" or "VLDB" venues).
- **Running Pipeline**: Execute `main.py` with ER configurations.
- **Configuration Options**:
  - `BLOCKING_METHODS`: Methods to reduce comparison space ["Year", "TwoYear", "numAuthors", "FirstLetter"].
  - `MATCHING_METHODS`: Algorithms for record comparison ["Jaccard", "Combined"].
  - `threshold`: A value between 0-1 for decision making.

**Selected Configuration**:
- Blocking method: `FirstLetter`
- Matching method: `Combined`
- Threshold: `0.5`

**Results**:
- The results are saved in `clustering_results.csv` within the "results" folder.

## PySpark Folder
The PySpark folder contains `DPREP.py` to compare the results with the Apache Spark framework.

## Data Folder 
The data folder includes the prepared and cleaned datasets and additional samples:

- `citation-acm-v8_1995_2004.csv`: ACM citation network dataset.
- `dblp_1995_2004.csv`: DBLP citation network dataset.
- `DIA_2023_Exercise.pdf`: Project instruction file.

**Note**: Check `requirements.txt` for compatibility before running the code.

---



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
