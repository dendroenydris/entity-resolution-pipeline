###################################################################################################
# Purpose: This script performs as a sample to reproduce our results
# Please install the necessary packages before using this script
# ```shell
# cd path-to-this-project
# pip install -e .
# ```
###################################################################################################
import logging
from erp import part1, part2, part3, naive_DPvsLocal
from erp.utils import FILENAME_DP_MATCHED_ENTITIES,FILENAME_LOCAL_MATCHED_ENTITIES
logging.basicConfig(level=logging.INFO, format="%(message)s")
part1() # cleaned data stored in "data" as "data/citation-acm-v8_1995_2004.csv" and "data/dblp_1995_2004.csv"
part2() # results of all methods stored in "results/method_results.csv"
part3() # scability test results stored in "results/scability_results.csv" and "results/scability.png"
naive_DPvsLocal(FILENAME_DP_MATCHED_ENTITIES,FILENAME_LOCAL_MATCHED_ENTITIES) # DP vs local results is printed in terminal