databases = []
columns = ["paper ID", "paper title", "author names", "publication venue", "year of publication"]

def prepare(file_name):

    # paper ID, paper title, author names, publication venue, and year of publication
    with open(file_name) as f:

        # Split the file into a list of entries
        list_of_files = f.read().split("\n\n")

        # Initialize a dataset with column length as requested schema, row length as number of entries
        databases = [['' for j in range(5)] for i in range(len(list_of_files))]

        # Define the order of information in each entry
        identify_string = ["#index", "#*", "#@", "#c", "#t"]

        # Split each entry into lines and extract information based on identify_string
        list_of_files = [col.split("\n") for col in list_of_files]
        for i in range(len(list_of_files)):
            for col in list_of_files[i]:
                for j in range(len(identify_string)):
                    if (col[:len(identify_string[j])] == identify_string[j]):
                        databases[i][j] = col[len(identify_string[j]):]
                        break

    # Filter the data based on specified criteria
    databases = [data for data in databases if data[-1] != '']
    databases = [data for data in databases if (
        (int(data[-1]) >= 1995) and (int(data[-1]) <= 2004) and (('SIGMOD' in data[-2]) or ('VLDB' in data[-2])))]

    # Write the filtered data to a CSV file
    with open(file_name[:-4]+"_1995_2004.csv", 'w') as f:
        f.write(
            "paper ID;;paper title;;author names;;publication venue;;year of publication;;\n")
        for cols in databases:
            line = ""
            for col in cols:
                line = line+col+';;'
            f.write(line+'\n')

if __name__ == "__main__":
    # Process the function on our two data files !
    for data in ["data/citation-acm-v8.txt", "data/dblp.txt"]:
        prepare(data)
