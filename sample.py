###################################################################################################
# Purpose: This script performs as a sample to reproduce our results
# Please install the necessary packages before using this script
# ```shell
# cd path-to-this-project
# pip install -e .
# pyspark --packages graphframes:graphframes:0.8.3-spark3.5-s_2.12
# ```
# spark == 3.5.0
# scala == 2.12.x
# graphframes == 0.8.3-spark3.5-s_2.12
###################################################################################################
import logging
from erp import clustering, part1, part2, part3, naive_DPvsLocal
from erp.main import ER_pipline
from erp.utils import (
    FILENAME_DP_MATCHED_ENTITIES,
    FILENAME_LOCAL_MATCHED_ENTITIES,
    DATABASES_LOCATIONS,
    DefaultERconfiguration,
)
from erp.dperp import DP_ER_pipline

logging.basicConfig(level=logging.INFO, format="%(message)s")
part1()  # cleaned data stored in "data" as "data/citation-acm-v8_1995_2004.csv" and "data/dblp_1995_2004.csv"
part2()  # results of all methods stored in "results/method_results.csv"
part3()  # scability test results stored in "results/scability_results.csv" and "results/scability.png"
ER_pipline(
    DATABASES_LOCATIONS[0], DATABASES_LOCATIONS[1], DefaultERconfiguration
)  # test the entire local pipline including clustering
DP_ER_pipline(
    DATABASES_LOCATIONS[0], DATABASES_LOCATIONS[1]
)  # test the entire parallel pipline including clustering
