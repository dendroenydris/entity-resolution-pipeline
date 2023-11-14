import numpy as np
import pandas as pd

databases = []
columns = ["paper ID", "paper title", "author names", "publication venue", "year of publication"]

def prepare(file_name):
    # paper ID, paper title, author names, publication venue, and year of publication
    with open(file_name) as f:
        list_of_files = f.read().split("\n\n")
        databases = [['' for j in range(5)] for i in range(len(list_of_files))]
        list_of_files = [col.split("\n") for col in list_of_files]
        identify_string = ["#index", "#*", "#@", "#c", "#t"]
        for i in range(len(list_of_files)):
            for col in list_of_files[i]:
                for j in range(len(identify_string)):
                    if (col[:len(identify_string[j])] == identify_string[j]):
                        databases[i][j] = col[len(identify_string[j]):]
                        break

    databases = [data for data in databases if data[-1] != '']
    databases = [data for data in databases if (
        (int(data[-1]) >= 1995) and (int(data[-1]) <= 2004) and (('SIGMOD' in data[-2]) or ('VLDB' in data[-2])))]

    with open(file_name[:-4]+"_1995_2004.csv", 'w') as f:
        f.write(
            "paper ID; paper title; author names; publication venue; year of publication;\n")
        for cols in databases:
            line = ""
            for col in cols:
                line = line+col+'; '
            f.write(line+'\n')

for data in ["data/citation-acm-v8.txt", "data/dblp.txt"]:
    prepare("data/dblp.txt")
