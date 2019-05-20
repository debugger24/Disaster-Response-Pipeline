import sys
import pandas as pd
import numpy as np
import sqlalchemy

def load_data(messages_filepath, categories_filepath):
    '''
        Load data from csv file
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    df = messages.merge(categories, on="id")
    
    return df

def clean_data(df):
    '''
       Return cleaned dataframe
    '''
    categories = df['categories'].str.split(';', expand=True)
    categories.columns = [item.split('-')[0] for item in df['categories'][0].split(';')]
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x.split('-')[1])
        
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    
    categories['related'] = categories['related'].apply(lambda x: 1 if x>0 else 0)
    df = pd.concat([df.drop(labels="categories", axis=1), categories], axis=1)
    df_unique = df.drop_duplicates(subset="message")
    return df_unique


def save_data(df, database_filename):
    '''
        Save data to SQLite Database
    '''
    engine = sqlalchemy.create_engine('sqlite:///' + database_filename)
    df.to_sql('messages', engine, index=False)  


def main():
    '''
        Main function to load, clean and save data
    '''
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