import sys
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords','averaged_perceptron_tagger'])
# import libraries
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import re
import pickle

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import confusion_matrix,classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier

import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

def load_data(database_filepath):
    
    """Connect to the clean sql database generated with process_data.py and split it into feature data (X) and response (Y)
    Args:
        database_filepath (str): string filepath of the sqlite database
    Returns:
        X (pandas dataframe): Feature data, just the messages
        Y (pandas dataframe): Classification labels
        category_names (list): List of the category names for classification
    """
    
    # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('df_clean',engine)
    df.replace([None], np.nan, inplace=True)
    df.dropna(inplace=True)
    X = df['message'] 
    Y =df[df.columns[4:]]
    Y['related']=Y['related'].map(lambda x: 1 if x == 2 else x)
    global category_names
    #category_names = Y.columns.values.tolist()
    category_names = Y.columns.tolist()
    return X,Y,category_names



def tokenize(text):
    
    """ A function to preprocess data:
            1.Detect and clean url's
            2.Remove special characters
            3.Tokenize Text and remove_stopwords
            4.lemmatize_text
            
        Returns:
            Clean and preprocessed text 
    """ 
    
    default_stopwords = set(stopwords.words("english")) 
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
        
    words = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    #tokens = nltk.word_tokenize(words)
    tokens = [w for w in nltk.word_tokenize(words) if w not in default_stopwords]
    lemmatizer = nltk.WordNetLemmatizer()
    tokens=[lemmatizer.lemmatize(w).strip() for w in tokens]
    return tokens


def build_model():
    
    """Returns the GridSearchCV object to be used as the model
    Args:
        None
    Returns:
        cv (scikit-learn GridSearchCV): Grid search model object
    """
   
    pipeline = Pipeline([

                    ('text_pipeline', Pipeline([
                        ('vect', CountVectorizer(tokenizer=tokenize)),
                        ('tfidf', TfidfTransformer())
                        ])),

                    ('clf', MultiOutputClassifier(AdaBoostClassifier()))
                        ])
    

    parameters={  
    'text_pipeline__vect__max_df': (.75, 1.0),
    'text_pipeline__tfidf__use_idf': (True, False),
    'clf__estimator__learning_rate':[.1,1,2]}
    
    cv = GridSearchCV(pipeline, param_grid=parameters, cv=3, n_jobs=-1, verbose=1)

    return cv
        


def evaluate_model(model, X_test, Y_test, category_names):
    """dumps the model to the given filepath
    Args:
        model (scikit-learn model): The fitted model
        model_filepath (string): the filepath to save the model to
    Returns:
        None
    """
    
 
    #category_names=Y_test.columns.values
    Y_pred_test = model.predict(X_test)        
    print(classification_report(Y_test.values, Y_pred_test, target_names=category_names))




def save_model(model, model_filepath):
    """dumps the model to the given filepath
    Args:
        model (scikit-learn model): The fitted model
        model_filepath (string): the filepath to save the model to
    Returns:
        None
    """
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