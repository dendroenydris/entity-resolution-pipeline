import logging
import pandas as pd

from erp.utils import DATABSE_COLUMNS, logging_delimiter

def prepare_data(file_name:str):
    """extracting and cleaning data

    Args:
        file_name (str): database txt file
    """
    databases = []
    logging_delimiter()
    logging.info(f"Preparing database: {file_name}")
    # paper ID, paper title, author names, publication venue, and year of publication
    with open(file_name) as f:
        # Split the file into a list of entries
        list_of_papers = f.read().split("\n\n")
        
        # Initialize a dataset with column length as requested schema, row length as number of entries
        databases = [["" for j in range(5)] for i in range(len(list_of_papers))]

        # Define the order of information in each entry
        identify_string = ["#index", "#*", "#@", "#c", "#t"]

        # Split each entry into lines and extract information based on identify_string
        list_of_papers = [col.split("\n") for col in list_of_papers]
        # for i in range(len(list_of_papers)):
        for i in range(len(list_of_papers)):
            if i % 100000 == 0:
                logging.info(f"Progress: {i / len(list_of_papers) * 100 :.2f}%")

            for col in list_of_papers[i]:
                for j in range(len(identify_string)):
                    filtered_string = "".join(
                        [
                            char
                            for char in col[len(identify_string[j]) :].lower()
                            if (char.isalnum()) or (char in [" ", ","])
                        ]
                    )

                    if col[: len(identify_string[j])] == identify_string[j]:
                        # Lowercase all characters and eliminate special characters
                        databases[i][j] = filtered_string
                        break

    # Filter the data based on specified criteria
    databases = [data for data in databases if (data[-1] != "") and (data[0] != "")]
    databases = [
        data
        for data in databases
        if (
            (int(data[-1]) >= 1995)
            and (int(data[-1]) <= 2004)
            and (("SIGMOD".lower() in data[-2]) or ("VLDB".lower() in data[-2]))
        )
    ]
    database = pd.DataFrame(databases, columns=DATABSE_COLUMNS)
    # Assuming in the same databse if the ids are the same, then the papers are the same, delete the duplicate ids.
    database = database.drop_duplicates(subset=["paper ID"])
    # Write the filtered data to a CSV file
    database.to_csv(file_name[:-4] + "_1995_2004.csv", index=False)
    logging.info(f"Cleaned data is stored in {file_name[:-4] + '_1995_2004.csv'}")
    logging_delimiter()

