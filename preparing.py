import pandas as pd

databases = []
columns = [
    "paper ID",
    "paper title",
    "author names",
    "publication venue",
    "year of publication",
]


def prepare_data(file_name):
    # paper ID, paper title, author names, publication venue, and year of publication
    with open(file_name) as f:
        # Split the file into a list of entries
        list_of_papers = f.read().split("\n\n")

        # Assuming in the same databse if the ids are the same, then the papers are the same, delete the duplicate ids.
        Lidx = []

        # Initialize a dataset with column length as requested schema, row length as number of entries
        databases = [["" for j in range(5)] for i in range(len(list_of_papers))]

        # Define the order of information in each entry
        identify_string = ["#index", "#*", "#@", "#c", "#t"]

        # Split each entry into lines and extract information based on identify_string
        list_of_papers = [col.split("\n") for col in list_of_papers]
        # for i in range(len(list_of_papers)):
        for i in range(len(list_of_papers)):
            if i % 10000 == 0:
                print("\b\r%.2f" % (i / len(list_of_papers) * 100), "%")

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

    # Write the filtered data to a CSV file
    with open(file_name[:-4] + "_1995_2004.csv", "w") as f:
        f.write(
            "paper ID;paper title;author names;publication venue;year of publication\n"
        )
        # Assuming in the same databse if the ids are the same, then the papers are the same, delete the duplicate ids.
        Lidx = []
        for cols in databases:
            line = ""
            if cols[0] in Lidx:
                continue
            else:
                Lidx.append(cols[0])
            for col in cols:
                line = line + col + ";"
            f.write(line[:-1] + "\n")


if __name__ == "__main__":
    # Process the function on our two data files !
    # prepare_data("data/test.txt")
    for data in ["data/citation-acm-v8.txt", "data/dblp.txt"]:
        prepare_data(data)
