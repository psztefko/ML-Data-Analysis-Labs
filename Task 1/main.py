import math

import pandas as pd
import numpy as np

# only for testing purposes
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def generate_input():
    return pd.DataFrame(np.random.randint(0, 100, size=(15, 4)), columns=list('ABCD'))


def min_max_column(column, min_value, max_value):
    new_column = []
    for current_number in column:
        x = (current_number - min_value) / (max_value - min_value)
        new_column.append(round(x, 2))
    return new_column


def normalize(df):
    min_value = df.min().min()
    max_value = df.max().max()
    for column in df:
        df[column] = min_max_column(df[column], min_value, max_value)
    return df


def calculate_nominator(df, df_mean):
    nominator = 0
    print(df)
    for column in df:
        for value in df[column]:
            nominator += pow(value - df_mean, 2)
    return nominator


def calculate_standard_deviation(df, df_mean):
    nominator = calculate_nominator(df, df_mean)
    return math.sqrt(nominator / df.count().count())


def standardize_column(df, column, df_mean):
    new_column = []
    standard_deviation = calculate_standard_deviation(df, df_mean)
    for current_number in column:
        x = (current_number - df_mean) / standard_deviation
        new_column.append(round(x, 2))
    return new_column


def standardize(df):
    df_mean = round(df.mean().mean(), 2)
    for column in df:
        df[column] = standardize_column(df, df[column], df_mean)
    return df


def test_min_max():
    df = generate_input()
    normalized_df = normalize(df)
    scaler = MinMaxScaler()
    sklearn_df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    df_diff = pd.concat([sklearn_df.round(2), normalized_df]).drop_duplicates(keep=False)

    print(f"Original dataframe\n {df} \n-------\n")
    print(f"Min-max normalized dataframe\n {normalized_df} \n-------\n")
    print(f"Sklearn normalized dataframe\n {sklearn_df} \n-------\n")
    print(f"Differences\n {df_diff} \n-------\n")


def test_standardization():
    df = generate_input()
    standardized_df = standardize(df)
    scaler = StandardScaler()
    scaler.fit(df)
    scaler.transform(df)
    df_diff = pd.concat([df.round(2), standardized_df]).drop_duplicates(keep=False)

    print(f"Original dataframe\n {df} \n-------\n")
    print(f"Min-max normalized dataframe\n {standardized_df} \n-------\n")
    print(f"Sklearn normalized dataframe\n {sklearn_df} \n-------\n")
    print(f"Differences\n {df_diff} \n-------\n")


if __name__ == '__main__':
    df = generate_input()
    # normalized_df = normalize(df)
    # standardized_df = standardize(df)

    # print(df)
    # print("------")
    # print(normalized_df)
    # print("------")
    # print(standardized_df)
    # print("------")

    # test_min_max()
    test_standardization()
