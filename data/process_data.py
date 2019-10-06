'''
Import all necessary packages for this file
Pandas, Numpy and sqlalchemy
'''
import sys
import pandas as pd
import numpy as np

from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    Function to load the data into dataframe from csv files
    - read csv files
    - merge dataframes

    Arguments: filepath of both messages and categories csv files

    Return: merged dataframe df of both files

    '''
    # Load both data sets
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # Merge datasets
    df = messages.merge(categories, on = "id", how = "inner")

    return df


def clean_data(df):
    '''
    Function to clean the data
    - essentially process the categories information which is contained
    in a single string per row into workable information
    - remove duplicates

    Argument: dataframe df to be cleaned

    Return: cleaned dataframe df

    '''

    # 1. Transform and clean the categories column
    # Extract categories
    cat = df['categories'].str.split(";", expand = True)

    # Extract 1st row to get categories names
    row = cat.iloc[0]

    # Get names of categories to be used as columns in dataframe
    cat_colnames = [row[i].split("-")[0] for i in range(len(row))]

    # Apply names to cat dataframe
    cat.columns = cat_colnames

    # Convert category values in numbers
    for col in cat:
        # get the last character of the string and set it as value
        cat[col] = pd.Series(cat[col]).astype(str).str[-1]
        # transform the string value into a number
        cat[col] = cat[col].astype(int)

    # replace categories column in dataframe with new cleaned category columns
    # remove initial column
    df.drop(['categories'], axis = 1, inplace = True)

    # add new cleaned categories columns
    df = pd.concat([df, cat], axis =1)

    # 2. Remove duplicates
    df = df.drop_duplicates()

    return df


def save_data(df, database_filename):
    '''
    Function to save the data into a SQL database

    Arguments:
        df: dataframe to be saved
        database_filename: the filename under which the database to be saved

    Return: none, just saves the file / database
    '''

    # create the engine
    engine = create_engine('sqlite:///'+ database_filename)

    # save the dataframe df into the database
    df.to_sql('table', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
