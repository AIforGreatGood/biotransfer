import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

##Load in csv data
def load_csv(csv_filename="R2R3_enrichment_replicates_I_II.csv"):
    df = pd.read_csv(csv_filename)
    df.rename(columns={df.columns[0]:"sequences"}, inplace = True)
    return df

##Split into train and val+test as specified in paper. train+val is all values contained in rep 1, test is all values contained in rep 2 that where not in train
def split_train_val_test(df):
    df_train_and_val = df.dropna(subset = ["log(R3/R2) replicate 1"])
    df_test = df[df['log(R3/R2) replicate 1'].isnull()].dropna(subset = ["log(R3/R2) replicate 2"])
    train_and_val_indices = df_train_and_val.index.tolist()
    test_indices = df_test.index.tolist()

    ##Sanity check to make sure all values are used. df_train_and_val + df_test = df
    master_list = train_and_val_indices + test_indices
    master_list.sort()

    assert(master_list == list(range(len(master_list))))

    ##Split train into train and val, 90/10 split
    df_train, df_val = train_test_split(df_train_and_val, test_size=0.1, random_state=314)

    return df_train, df_val, df_test

def convert2json(df, target, output_filename):
    df.rename(columns={target:"target"}, inplace=True)
    json_text = df[['sequences','target']].to_dict('records')
    with open(output_filename, 'w') as json_file:
        json.dump(json_text, json_file)

def get_histogram(df, column, output_filename):
    histogram = df.hist(bins=50,column=column)[0][0]
    figure = histogram.get_figure()
    plt.title(os.path.basename(output_filename).split('.')[0].replace("_"," "))
    figure.savefig(output_filename)
    return histogram

def get_stats(df, column, output_filename):
    print('Column Counts:')
    print(df.count())
    print('Replicate Value Counts for Ranges:')
    print(df[column].value_counts(bins=10))
    get_histogram(df, column, output_filename)



def main():
    df = load_csv()
    df_train, df_val, df_test = split_train_val_test(df)

    print('Train')
    get_stats(df_train, "log(R3/R2) replicate 1", "Gifford_Train.png")

    print("Val")
    get_stats(df_val, "log(R3/R2) replicate 1", "Gifford_Val.png")

    print("Test")
    get_stats(df_test, "log(R3/R2) replicate 2","Gifford_Test.png")
    
    convert2json(df_train, "log(R3/R2) replicate 1","gifford_train.json")
    convert2json(df_val, "log(R3/R2) replicate 1", "gifford_val.json")
    convert2json(df_test, "log(R3/R2) replicate 2", "gifford_test.json")

main()
