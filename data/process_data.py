"""
Project: Disaster Response Pipeline

ETL pipeline

The pipeline is responsible for loading data from CSV files, cleaning data and saving the processed data to SQLite database.

Run the following command in project directory to execute the ETL pipeline:
    
Python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/disaster_response_db.db

"""


import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
 
# Function for loading massages and categories data
def load_data(messages_filepath, categories_filepath):

    """
    Inputs:
        messages_filepath:  Path to the CSV file containing messages
        categories_filepath: Path to the CSV file containing categories
    
    Output:
        df: Merged messages and categories
    """
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages,categories,on='id')
    
    return df 

# Function for cleaning data
def clean_data(df):
    
    """    
    Input:
        df: Combined messages and categories
    
    Output:
        df: Cleaned data with new column for each categories.
    """
    
    # Split the categories
    categories = df['categories'].str.split(pat=';',expand=True)
    
    #Fix the categories columns name
    row = categories.iloc[[1]]
    category_colnames = [category_name.split('-')[0] for category_name in row.values[0]]
    categories.columns = category_colnames
    
    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = categories[column].astype(np.int)
    
    df = df.drop('categories',axis=1)
    df = pd.concat([df,categories],axis=1)
    df = df.drop_duplicates()
    
    return df

# Function for saving data to SQL database
def save_data(df, database_filename):
    
    """
    Inputs:
        df: Cleaned messages and categories data 
        database_filename: Path to SQLite destination database
    """
    
    engine = create_engine('sqlite:///'+ database_filename)
    table_name = "processeddata" + "_table"
    df.to_sql(table_name, engine, index=False, if_exists='replace')



def main():
    """
    Main function perfroms following tasks: 
        1) Loading Messages and Categories
        2) Cleaning Categories 
        3) Saving Data to SQLite Database
    """
    
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:] 

        print('Loading messages data from {} ...\nLoading categories data from {} ...'
              .format(messages_filepath, categories_filepath))
        
        df = load_data (messages_filepath, categories_filepath)

        print('Cleaning data ...')
        df = clean_data (df)
        
        print('Saving data to SQLite DB : {}'.format(database_filepath))
        save_data (df, database_filepath)
        
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