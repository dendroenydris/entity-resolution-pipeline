import logging
import random
from time import time
import numpy as np
import pandas as pd
from erp.dperp import DP_ER_pipline
from erp.matching import *
from erp.clustering import clustering_basic, clustering
from erp.preparing import prepare_data
from erp.utils import (
    FILENAME_DP_LOCAL_DIFFERENCE,
    FILENAME_LOCAL_CLUSTERING,
    FILENAME_LOCAL_MATCHED_ENTITIES,
    FILENAME_DP_MATCHED_ENTITIES,
    FILENAME_SCABILITY_TEST_RESULTS,
    RESULTS_FOLDER,
    test_and_create_folder,
    DefaultERconfiguration,
    DATABSE_COLUMNS,
    DATABASES_LOCATIONS,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
import string
import matplotlib.pyplot as plt


def ER_pipline(
    dfilename1,
    dfilename2,
    ERconfiguration,
    baseline=False,
    matched_output=FILENAME_LOCAL_MATCHED_ENTITIES,
    cluster_output=FILENAME_LOCAL_CLUSTERING,
    cluster=True,
):
    # import database
    df1 = pd.read_csv(dfilename1, sep=",", engine="python")
    df2 = pd.read_csv(dfilename2, sep=",", engine="python")

    df1["index"] = np.arange(len(df1))
    df2["index"] = np.arange(len(df2)) + len(df1)

    similarity_threshold = ERconfiguration["threshold"]
    start_time = time()
    result_df = blocking(df1, df2, ERconfiguration["blocking_method"])
    result_df = matching(
        result_df,
        similarity_threshold,
        ERconfiguration["matching_method"],
        outputfile=matched_output,
    )
    end_time = time()
    matching_time = end_time - start_time
    c_df = []
    if cluster:
        c_df = clustering(
            result_df,
            df1,
            df2,
            ERconfiguration["clustering_method"],
            filename=cluster_output,
        )
    end_time = time()

    if baseline:
        baseline_df = calculate_baseline(
            df1,
            df2,
            baseline_config={
                "method": ERconfiguration["matching_method"],
                "threshold": ERconfiguration["threshold"],
            },
        )

        return resultToString(
            ERconfiguration,
            -1,
            -1,
            -1,
            baseline_df,
            matched_df=result_df,
            suffix="_local",
        )
    return {
        "local rate": round(len(c_df) / (len(df1) + len(df2)), 4),
        "local execution time": round((end_time - start_time) / 60, 2),
        "local execution time(matching+blocking)": round(matching_time / 60, 2),
    }


def add_random_characters_to_string(str, number):
    characters = string.ascii_lowercase
    new_string = list(str)

    for _ in range(number):
        random_char = random.choice(characters)
        new_string[random.randint(0, len(str) - 1)] = random_char

    return "".join(new_string)


DATABASE_CHANGES_CHOICE = ["year", "author", "title"]


def databaseWithMinimalChanges(filename, option="title", num=3):
    """minimal modification to database

    Args:
        database (_type_): _description_
        option (str, optional): "year"|"author"|"title".
        num (int): number of changes
    """
    database = pd.read_csv(filename)
    if option == "author":
        database[DATABSE_COLUMNS[2]] = database.apply(
            lambda x: add_random_characters_to_string(str(x[DATABSE_COLUMNS[2]]), num),
            axis=1,
        )
    elif option == "title":
        database[DATABSE_COLUMNS[1]] = database.apply(
            lambda x: add_random_characters_to_string(str(x[DATABSE_COLUMNS[1]]), num),
            axis=1,
        )
    elif option == "year":
        database[DATABSE_COLUMNS[-1]] += random.randint(-int(num / 2), -int(num / 2))
    return database


def create_databaseWithChanges(L_filename, num=3, cnum=3):
    L_datanames = []
    for filename in L_filename:
        for i in range(num):
            option = DATABASE_CHANGES_CHOICE[i % len(DATABASE_CHANGES_CHOICE)]
            df = databaseWithMinimalChanges(filename, option, cnum)
            new_filename = filename[5:-4] + "_" + option[:4] + str(cnum) + ".csv"
            new_folder = filename[:-4]
            test_and_create_folder(new_folder)
            df.to_csv(new_folder + "/" + new_filename)
            L_datanames.append(new_folder + "/" + new_filename)
    return L_datanames


def plot_scability_figures(results):
    # Assuming results is a DataFrame or a dictionary containing the required data
    # results = {"d1-d2": [...], "dp execution time": [...], "local execution time": [...]}

    # Convert "d1-d2" values to strings
    results["d1-d2"] = [str(i) for i in results["d1-d2"]]

    # Set the width of the bars
    bar_width = 0.35

    # Create an array of indices for the first set of bars
    indices = np.arange(len(results["d1-d2"]))

    # Plot the first set of bars
    plt.figure(figsize=(10, 4), dpi=100)
    plt.bar(
        indices,
        results["dp execution time"],
        label="dp execution time",
        alpha=0.6,
        width=bar_width,
    )

    # Shift the indices for the second set of bars
    indices_shifted = indices + bar_width

    # Plot the second set of bars
    plt.bar(
        indices_shifted,
        results["local execution time"],
        label="local execution time",
        alpha=0.6,
        width=bar_width,
    )

    plt.xlabel("replication factor")
    plt.ylabel("execution time(min)")
    plt.title("Resulting Execution Time")
    plt.xticks(indices + bar_width / 2, results["d1-d2"], fontsize=7)
    plt.legend(loc="upper right")
    plt.savefig(RESULTS_FOLDER + "scability.png")


def part1():
    for data in DATABASES_LOCATIONS:
        prepare_data(data)


def part2(thresholds=[0.5, 0.7]):
    # import database
    df1 = pd.read_csv(DATABASES_LOCATIONS[0], sep=",", engine="python")
    df1["index"] = np.arange(len(df1))
    df2 = pd.read_csv(DATABASES_LOCATIONS[1], sep=",", engine="python")
    df2["index"] = np.arange(len(df2)) + len(df1)
    # Run all blocking methods for each baseline and record results
    run_all_blocking_matching_methods(
        df1, df2, thresholds, MATCHING_METHODS, BLOCKING_METHODS
    )


def scability_test(
    ERconfiguration=DefaultERconfiguration,
    num_duplicates=3,
    num_changes=4,
    output=FILENAME_SCABILITY_TEST_RESULTS,
):
    L_filenames = create_databaseWithChanges(
        DATABASES_LOCATIONS, num_duplicates, num_changes
    )
    D = [(d1, d2) for d1 in L_filenames[:3] for d2 in L_filenames[3:]]
    results = []

    for d1, d2 in D:
        result = ER_pipline(d1, d2, ERconfiguration, baseline=False, cluster=False)
        result["d1-d2"] = (d1[-9:-4], d2[-9:-4])
        result2 = DP_ER_pipline(d1, d2, DefaultERconfiguration, cluster=False)
        results.append({**result2, **result})
    results = pd.DataFrame(results)
    save_result(results, output)
    plot_scability_figures(results)


def part3():
    scability_test()
    naive_DPvsLocal(
        FILENAME_DP_MATCHED_ENTITIES, FILENAME_LOCAL_MATCHED_ENTITIES
    )  # DP vs local results is printed in terminal


def compareTwoDatabase(df1, df2, name_df1="df1", name_df2="df2", item_name="matched pairs"):
    tp, fn, fp, precision, recall, f1 = calculate_confusion_matrix(df1, df2)
    merged = pd.merge(df1, df2, how="outer", indicator=True)
    differences = merged[merged["_merge"] != "both"]
    differences.to_csv(RESULTS_FOLDER + FILENAME_DP_LOCAL_DIFFERENCE)

    # print the number of {item_name} in each DataFrame
    logging.info(f"Number of {item_name} in {name_df1}: {len(df1)}")
    logging.info(f"Number of {item_name} in {name_df2}: {len(df2)}")

    # print the results
    logging.info(f"Number of differences: {fn+fp}")
    logging.info(f"Number of shared elements: {tp}")
    logging.info(f"Number of elements in {name_df1} but not in {name_df2}: {fn}")
    logging.info(f"Number of elements in {name_df2} but not in {name_df1}: {fp}")


def naive_DPvsLocal(fdp, flocal):
    DP_ER_pipline(
        DATABASES_LOCATIONS[0],
        DATABASES_LOCATIONS[1],
        DefaultERconfiguration,
        matched_output=fdp,
        cluster=False,
    )
    ER_pipline(
        DATABASES_LOCATIONS[0],
        DATABASES_LOCATIONS[1],
        ERconfiguration=DefaultERconfiguration,
        matched_output=flocal,
        cluster=False,
    )
    df_dp = pd.read_csv(RESULTS_FOLDER + fdp)
    df_local = pd.read_csv(RESULTS_FOLDER + flocal)
    compareTwoDatabase(df_dp, df_local, name_df1="DP ERP", name_df2="local ERP")
