'''
Import all necessary libraries
'''

import sys
import pickle
from sqlalchemy import create_engine
import pandas as pd

# NLP Packages
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# ML packages
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def load_data(database_filepath):
    '''
    Function to load the database into a dataframes

    Argument: filepath of database

    Return: TBD
    X, Y, category_names = load_data(database_filepath)
    '''

    # create engine
    engine = create_engine('sqlite:///'+ database_filepath)

    # load db into dataframe df
    df = pd.read_sql_table('table', con = engine)

    # Extract X, Y and category_names from dataframe
    X = df['message']
    Y = df.iloc[:,4:]
    category_names = Y.columns

    # return X, Y and category_names
    return X, Y, category_names


def tokenize(text):
    '''
    Function to tokenize the text, e.g. get a text format that can be processed

    Argument: text to be tokenized

    Return: tokenized text
    '''

    # Normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # tokenize text
    tokens = word_tokenize(text)

    # lemmatize and remove stop words
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords.words('english')]

    return tokens


def build_model():
    '''
    Function to build a machine learning model for this used

    Argument: none

    Return: model
    '''

    # Instantiate an adabosst
    ada = AdaBoostClassifier()

    # Build a pipeline
    model = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(ada))
        ])

    return model

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Function to evaluate the model using the test set

    Arguments:
    - model: model to be evaluated
    - X_test, Y_test: test set for evaluation
    - category_names: categories for which the model is to be evaluated

    Return: none, prints evaluation results
    '''

    # Using model, predict from test set
    Y_pred = model.predict(X_test)

    # Transform Y_pred into dataframe
    Y_pred_df = pd.DataFrame(Y_pred, columns = Y_test.columns)

    # Print classification report
    for cat in category_names:
        print('Model performance with category: {}'.format(cat))
        print(classification_report(Y_test[cat], Y_pred_df[cat]))

    # Print overall accuracy
    # print("Overall accuracy; {}".format((Y_pred_df == Y_test).mean().mean()))


def save_model(model, model_filepath):
    '''
    Function to save the model using pickle

    Arguments:
    - model: to be saved
    - model_filepath: filepath under which the model is to be saved

    Return: none, model saved
    '''

    pickle.dump(model, open(model_filepath, 'wb'))


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
