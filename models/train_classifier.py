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
nltk.download('averaged_perceptron_tagger')

# ML packages
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.base import BaseEstimator, TransformerMixin

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
    df = pd.read_sql_table('df', con = engine)

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

    # Find ulr in text and replace with a urlplaceholder
    ## Define form of url
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    ## find all urls in text
    detected_urls = re.findall(url_regex, text)

    ## replace url with urlplaceholder
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    # normalize case and remove punctuation
    # text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # tokenize text
    tokens = word_tokenize(text)

    # lemmatize and remove stop words
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

class TextLengthExtractor(BaseEstimator, TransformerMixin):
    '''
    Create a transformer to enrich the machine learning model

    This transformer extracts the length of the text message

    No input and output, just functions inside the class
    '''

    def fit(self, X, y = None):
        '''
        No need to fit as this is only a transformer
        '''
        return self

    def transform(self, X):
        '''
        Transform function to extract the length of the text
        '''
        return pd.DataFrame(pd.Series(X).apply(lambda x: len(x)))

class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    '''
    Create a transformer to enrich the machine learning model

    This transformer extracts if starting verb of the text message

    No input and output, just functions inside the class
    '''

    def starting_verb(self, text):
        '''
        Define function for the transform method

        Arg: Text

        Return: true if text starts with a verb, otherwise false
        '''

        sentence_list = nltk.sent_tokenize(text)

        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True

        return False

    def fit(self, x, y=None):
        '''
        No need to fit as this is only a transformer
        '''

        return self

    def transform(self, X):
        '''
        Transform method to run the starting_verb function
        '''

        X_tagged = pd.Series(X).apply(self.starting_verb)

        return pd.DataFrame(X_tagged)

def build_model():
    '''
    Function to build a machine learning model for this used

    Argument: none

    Return: model
    '''

    # Instantiate an adabosst
    ada = AdaBoostClassifier()

    # Build a pipeline
    # using feature union, adding some more features into the model
    model = Pipeline([
        ('features', FeatureUnion([

            ('nlp_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer = tokenize,
                        max_features = 10000, ngram_range = (1, 1),
                        max_df = 0.5)),
                ('tfidf', TfidfTransformer())
                ])),

            ('starting_verb', StartingVerbExtractor()),

            ('text_length', TextLengthExtractor())
                ])),

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
