# Entity Resolution Pipline

## Installation

The prerequisites in addition to the python packages(requirements.txt) are

> spark == 3.5.0
> scala == 2.12.x
> graphframes == 0.8.3-spark3.5-s_2.12

```shell
cd path-to-this-project
pip install -e .
```

## Samples

[part1/2/3](./sample.py)

```python
from erp import part1, part2, part3
part1() # cleaned data stored in "data"
part2() # results of all methods stored in "method_results.csv"
part3() # scability test
```

## Quick Start

- **Preparing Data**: Run `erp.preparing.prepare_data("path_to_txt_file")` for both text files. This will clean and extract the relevant data (1995-2004 citations by "SIGMOD" or "VLDB" venues). The resulting csv files will show in `data` folder.
- **Running Pipeline**:
  - Local Version : Run `erp.ER_pipeline(databasefilename1, databasefilename2, ERconfiguration, baseline=False, cluster=True,matched_output="path-to-output-file", cluster_output="path-to-output-file", isdp=False)` (in `erp/main.py`)
  - DP Version: Run `erp.ER_pipeline(databasefilename1, databasefilename2,  ERconfiguration, baseline=False, cluster=True, matched_output=F"path-to-output-file", cluster_output="path-to-output-file", isdp=True)` (it calls `ER_pipeline_dp` in `erp/dperp.py`)
- **Configuration Options**:
  - `blocking_method`(String): Methods to reduce execution time `{“Year”, “TwoYear”, “numAuthors”, “FirstLetterTitle”, “LastLetterTitle”, "FirstOrLastLetterTitle", “authorLastName”, “commonAuthors”, “commonAndNumAuthors”}`.
  - `matching_method`(String): Algorithms for entity matching `{"Jaccard", "Combined"}`.
  - `clustering_method`(String): Altogirthm for clustering {"basic"}.
  - `threshold`(float): A value between 0.0-1.0 for the matching similarity threshold.
  - `output_filename`(String): path and file name of clustering results to be saved.

### Selected Configuration**

ERconfiguration:

```json
{
  "matching_method": "Combined",
  "blocking_method": "FirstOrLastLetterTitle",
  "clustering_method": "basic",
  "threshold": 0.7,
  "output_filename": "clustering_results_local.csv"
}
```