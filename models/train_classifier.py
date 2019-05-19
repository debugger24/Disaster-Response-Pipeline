import sys

import pandas as pd
import numpy as np
import sqlalchemy

import re
import pickle

import nltk
nltk.download('punkt')
nltk.download('stopwords')

from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC

import warnings


def load_data(database_filepath):
    # load data from database
    engine = sqlalchemy.create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('messages', engine)
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre', 'child_alone'], axis=1)
    return X, Y, Y.columns


stop_words = stopwords.words("english")

def tokenize(text):
    # Convert to lower case 
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # Tokenize
    tokens = word_tokenize(text.lower())
    
    # Remove stop words
    tokens = [token for token in tokens if token not in stop_words]
    
    return tokens


def build_model():
    pipeline = Pipeline([
        ('count', CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer()),
        ('classifier', MultiOutputClassifier(SVC(kernel='linear', degree=2, C=1)))
    ])
    
    parameters = {
        'classifier__estimator__kernel': ['linear'],
        'classifier__estimator__degree':[2],
    }
    
    cv = GridSearchCV(pipeline, 
                      param_grid=parameters, 
                      verbose=3)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = pipeline.predict(X_test)
    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    for i in range(len(col_names)):
        accuracy = accuracy_score(y_pred[:,i], y_test[col_names[i]].values)
        precision = precision_score(y_pred[:,i], y_test[col_names[i]].values)
        recall = recall_score(y_pred[:,i], y_test[col_names[i]].values)
        f1 = f1_score(y_pred[:,i], y_test[col_names[i]].values)
        accuracy_list.append(accuracy)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
    df = pd.DataFrame({'label': col_names,
                       'accuracy': accuracy_list,
                       'precision': precision_list,
                       'recall': recall_list,
                       'f1': f1_list})
    print(df)


def save_model(model, model_filepath):
    pickle.dump(model, open('model_filepath', 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()