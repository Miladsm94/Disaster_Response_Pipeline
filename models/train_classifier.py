"""
Project: Disaster Response Pipeline

Classifier Training

The pipeline is responsible for loading data from SQLite database, processing the text and building classification pipeline.

Run the following command in project directory to execute the script:
    
Python train_classifier.py ../data/disaster_response_db.db classifier.pkl

"""

import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
import numpy as np
import pandas as pd
import sys
import os
import re
import pickle
from sqlalchemy import create_engine
from scipy.stats import gmean
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report,fbeta_score, make_scorer,confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.base import BaseEstimator,TransformerMixin

def load_data(database_filepath):
    """
    Load Data from the SQLite Database 
    
    Inputs:
        database_filepath 
    Output:
        X: dataframe containing features
        Y: dataframe containing labels
        category_names: List of categories name
    """
    
    engine = create_engine('sqlite:///' + database_filepath)
    
    table_name = "processeddata" + "_table"
    df = pd.read_sql_table(table_name,engine)
    
    # Remove column with only zero values.
    df = df.drop(['child_alone'],axis=1)
    
    X = df['message']
    y = df.iloc[:,4:]
    
    category_names = y.columns 
    return X, y, category_names


def tokenize(text,url_place_holder_string="urlplaceholder"):
    """
    Tokenize the text 
    
    Inputs:
        text: Text message 
    Output:
        clean_tokens: List of extracted tokens
    """
    
    # Replace all urls with a urlplaceholder string
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    # Extract all the urls from the provided text 
    detected_urls = re.findall(url_regex, text)
    
    # Replace url with a url placeholder string
    for detected_url in detected_urls:
        text = text.replace(detected_url, url_place_holder_string)

    # Extract the word tokens from the provided text
    tokens = nltk.word_tokenize(text)
    
    # Lemmatize the words
    lemmatizer = nltk.WordNetLemmatizer()

    # List of clean tokens
    clean_tokens = [lemmatizer.lemmatize(w).lower().strip() for w in tokens]
    return clean_tokens

# Build a custom transformer to extract the starting verb of sentences
class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """
    Starting Verb Extractor class
    
    This class creates a new feature by extracting the starting verb of the sentence
    """

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)

def build_pipeline():
    """
    Build Pipeline
    
    Output:
        A Scikit ML Pipeline that process text messages and apply a classifier.
        
    """
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('count_vectorizer', CountVectorizer(tokenizer=tokenize)),
                ('tfidf_transformer', TfidfTransformer())
            ])),

            ('starting_verb_transformer', StartingVerbExtractor())
        ])),

        ('classifier', MultiOutputClassifier(AdaBoostClassifier()))
    ])

    return pipeline

def multioutput_fscore(y_true,y_pred,beta=1):
    
    """
    Calculate FScore
        
    Inputs:
        y_true: List of labels
        y_prod: List of predictions
        beta: Beta value to be used to calculate fscore metric
    
    Output:
        f1score: Calculation geometric mean of fscore
    """
    
    # If "y_pred" is a dataframe then extract the values from that
    if isinstance(y_pred, pd.DataFrame) == True:
        y_pred = y_pred.values
    
    # If "y_true" is a dataframe then extract the values from that
    if isinstance(y_true, pd.DataFrame) == True:
        y_true = y_true.values
    
    f1score_list = []
    for column in range(0,y_true.shape[1]):
        score = fbeta_score(y_true[:,column],y_pred[:,column],beta,average='weighted')
        f1score_list.append(score)
        
    f1score = np.asarray(f1score_list)
    f1score = f1score[f1score<1]
    
    # Get the geometric mean of f1score
    f1score = gmean(f1score)
    return f1score

def evaluate_pipeline(pipeline, X_test, Y_test, category_names):
    
    """
    Evaluate the model
    
    This function applies a ML pipeline to a test set and prints out the model performance (accuracy and f1score)
    
    Inputs:
        pipeline: Scikit ML Pipeline
        X_test: Test features
        Y_test: Test labels
        category_names: label names 
    """
    Y_pred = pipeline.predict(X_test)
    
    multi_f1 = multioutput_fscore(Y_test,Y_pred, beta = 1)
    overall_accuracy = (Y_pred == Y_test).mean().mean()

    print('Average overall accuracy {0:.2f}%'.format(overall_accuracy*100))
    print('F1 score (custom definition) {0:.2f}%'.format(multi_f1*100))

    # Print the classification report.
    Y_pred = pd.DataFrame(Y_pred, columns = Y_test.columns)
    
    for column in Y_test.columns:
        print('Model Performance with Category: {}'.format(column))
        print(classification_report(Y_test[column],Y_pred[column]))


def save_model_as_pickle(pipeline, pickle_filepath):
    
    """
    Save Pipeline function
    
    This function saves trained model as Pickle file, to be loaded later.
    
    Inputs:
        pipeline: GridSearchCV or Scikit Pipelin object
        pickle_filepath: path for saving .pkl file
    """
    pickle.dump(pipeline, open(pickle_filepath, 'wb'))

def main():
    """
    Train Classifier 
    
    This function applies the Machine Learning Pipeline:
        1) Extract data from SQLite db
        2) Train ML model on training set
        3) Estimate model performance on test set
        4) Save trained model as .pkl file
    
    """
    if len(sys.argv) == 3:
        database_filepath, pickle_filepath = sys.argv[1:]
        print('Loading data from {} ...'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building the pipeline ...')
        pipeline = build_pipeline()
        
        print('Training the pipeline ...')
        pipeline.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_pipeline(pipeline, X_test, Y_test, category_names)

        print('Saving pipeline to {} ...'.format(pickle_filepath))
        save_model_as_pickle(pipeline, pickle_filepath)

        print('Trained model saved!')

    else:
         print("'Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl'")

if __name__ == '__main__':
    main()
