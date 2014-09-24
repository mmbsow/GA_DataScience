# -*- coding: utf-8 -*-
"""
Created on Sat Sep 20 15:10:07 2014

@author: Mariame
"""

import numpy as np
import pandas as pd
#from matplotlib import pyplot as plt
#from sklearn.cross_validation import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
#from sklearn.metrics import precision_score, recall_score, f1_score
#from sklearn.pipeline import Pipeline
#from sklearn.grid_search import GridSearchCV
from nltk import word_tokenize
from nltk.stem import PorterStemmer
import re
import sys
import getopt
import time
import datetime
import threading

_debug = True

_valid_classifiers = {
    "logistic": LogisticRegression,
    "randomforest": RandomForestClassifier,
}

# set seed for randomization
if _debug:
    np.random.seed(123)

def timestamp(date):
    return time.mktime(date.timetuple())


class SentAnaVectorizer(object):
    _count_vectorizer = None
    _vectors = {}
    
    def __init__(self, embeddings, binary=False):
        self._count_vectorizer = CountVectorizer(binary, stop_words='english')
        self.embeddings = embeddings
        
    def _ngram2vector(self, ngram):
        """
        Converts an n-gram phrase into a single brown vector by summing element-wise 
        the brown vectors of the individual words of the phrase
        """
        ngram_vect = [ [0 for i in range(1,51)] ]
        
#        terms_list = re.findall(r"[\w']+|[!\"#$%&\'()*+,-./:;<=>?@\[\\\]^_`{|}~]", str(ngram).lower())
        terms_list = re.findall(r"[\w']+", str(ngram).lower())
        for term in terms_list:
            term_vect = self.embeddings.get(term, [0 for i in range(1,51)])
            ngram_vect += [ term_vect ]
            
        return [ np.mean(x) for x in zip(*ngram_vect) ]

        
    def fit(self, X):
        # fits the normal CountVectorizer
        self._count_vectorizer.fit(X)
        
        # gets the list of brown vectors for each phrase in the train set
        for xi in X:
            self._vectors[xi] = self._ngram2vector(xi)
            
        if _debug:
            print 'Length of _embed_vect: ', len(self._vectors)
            #print self._vectors[0:5]
 
    def transform(self, X):
        # transform using count vectorizer
        X_sparse = self._count_vectorizer.transform(X)
        X_array = X_sparse.todense()
        
        # transform using embeddings
        vectors_list = []
        for xi in X:
            v = self._vectors.get(xi, None)
            if not v:
                v = self._ngram2vector(xi)
            vectors_list += [ v ]
        vectors_array = np.array(vectors_list)
        
        # concatenate both results
        return np.concatenate((X_array, vectors_array), axis=1)
        
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
       
       
class SentAnaWorker(threading.Thread):
    def __init__(self, train, test=None, embeddings=None):
        threading.Thread.__init__(self)
        
        self.train = train
        self.test = test
        self.embeddings = embeddings
        
    def setClassifier(self, classifier, args={}):
        self.classifier = _valid_classifiers[classifier](**args)
            
    def split_sets(self):
        X_train = self.train['Phrase']
        y_train = self.train['Sentiment']
        X_test, y_test = (None, None)
        
        # split train set if no test set provided
        if not self.test:
            from sklearn.cross_validation import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X_train, y_train)
        else:
            X_test = self.test['Phrase']
    
        if _debug:
            print 'Count in X_train: ', len(X_train)
            print 'Count in X_test: ', len(X_test)
        
        return (X_train, X_test, y_train, y_test)
    
    def run(self):
        """
        Starts the thread and run the model
        """
        X_train, X_test, y_train, y_test = self.split_sets()
        
        # gets the list of brown vectors for each phrase in the train set
        vectorizer = SentAnaVectorizer(self.embeddings, binary=False)
        X_train_v = vectorizer.fit_transform(X_train)
        if _debug:
            print 'Type of X_train_v: ', type(X_train_v)
            print 'Shape of X_train_v: ', X_train_v.shape

        # predict the y for the test data
        X_test_v = vectorizer.transform(X_test)
        if _debug:
            print 'Type of X_test_v: ', type(X_test_v)
            print 'Shape of X_test_v: ', X_test_v.shape
 
        # now run the classifier
        starttime = datetime.datetime.now()
        self.classifier.fit(X_train_v, y_train)
        endtime = datetime.datetime.now()
        print 'Logistic Run Time (secs): ', timestamp(endtime) - timestamp(starttime)
            
        predictions = self.classifier.predict(X_test_v)
        
        if y_test.any():
            if _debug:
                dispcount = 10 if len(y_test) > 10 else len(y_test)
                for i in range(dispcount):
                    print y_test[i], predictions[i]
                    
                print 'accuracy', self.classifier.score(X_test_v, y_test)
                print classification_report(y_test, predictions)

        # test Random Forest Classifier
#        starttime = datetime.datetime.now()
#        classifier2 = RandomForestClassifier(
#            n_estimators = int(math.ceil(X_test_v.shape[1] - 1)),
#            max_depth = None, # 50
#            min_samples_leaf = 1, #10
#            criterion='entropy',
#            n_jobs = -1,
#            bootstrap = False,
#            verbose = 1)
#        classifier2.fit(X_train_v, y_train)
#        endtime = datetime.datetime.now()
#        print 'RandomForest: Duration (secs): ', timestamp(endtime) - timestamp(starttime)
#    
#        predictions2 = classifier2.predict(X_test_v)
#        if y_test.any():
#            if _debug:
#                dispcount = 10 if len(y_test) > 10 else len(y_test)
#                for i in range(dispcount):
#                    print y_test[i], predictions2[i]
#                    
#            print 'accuracy', classifier2.score(X_test_v, y_test)
#            print classification_report(y_test, predictions2)
#                     
        
#    pipeline = Pipeline([
#        ('clf', RandomForestClassifier(criterion='entropy'))
#    ])
#
##    n_features = int(math.ceil(X_test_v.shape[1] - 1))
#    parameters = {
#        'clf__n_estimators': (5, 10),
#        'clf__max_depth': (50,100),
#        'clf__min_samples_split': (1, 2),
#        'clf__min_samples_leaf': (1, 2)
#    }
#
#    grid_search = GridSearchCV(pipeline, parameters, n_jobs=1, verbose=3, scoring='f1')
#    grid_search.fit(X_train_v, y_train)
#
#    print 'Best score:', grid_search.best_score_
#    print 'Best parameters set:'
#    best_parameters = grid_search.best_estimator_.get_params()
#    for param_name in sorted(parameters.keys()):
#        print '\t', param_name, best_parameters[param_name]
        
    
def main(**argv):
    '''
    Main function - entry point of the program
    Args:
        - argv: arguments passed from the command line
    '''
    trainfile = None
    testfile = None
    embedfile = None
    
    # read arguments from the command line
    try:
        opts, args = getopt.getopt(argv,"hfte:",["file=","test=","embed="])
    except getopt.GetoptError:
        print 'sentana.py -f <trainfile> -t <testfile> -e <embedfile> OR sentana.py'
        sys.exit(2)
        
    # process the arguments passed in te command line
    for opt, arg in opts:
        if opt == '-h':
            print 'sentana.py -f <trainfile> -t <testfile> -e <embedfile> OR sentana.py'
            sys.exit()
        elif opt in ("-f", "--file"):
            trainfile = arg
        elif opt in ("-t", "--test"):
            testfile = arg
        elif opt in ("-e", "--embed"):
            embedfile = arg
        
    # load the train data
    if not trainfile:
        trainfile = 'train.tsv'
    train = pd.read_table(trainfile)

    # if no test file passed, we will split the train set
    test = None
    if testfile:
        test = pd.read_table(testfile)
        
    if not embedfile:
        embedfile = 'embeddings.txt'
    embed_df = pd.read_table(embedfile, header=None, sep=' ', names=['word'] + range(1,51))
    embeddings = {row[0]: list(row[1:]) for row in embed_df.as_matrix()}
            
    if _debug:
        train = train#.head(10000)
        
    
    # use Logistic regression
    worker1 = SentAnaWorker(train, test, embeddings)
    worker1.setClassifier('logistic')
    worker1.start()
    
#    # use Random Forest Classifier
#    forest_args = {'n_estimators': int(math.sqrt(math.ceil(X_test_v.shape[1]))),
#                   'max_depth': None,
#                   'min_samples_leaf': 1,
#                   'criterion': 'entropy',
#                   'n_jobs': -1,
#                   'bootstrap': False,
#                   'verbose': 1}
#    worker2 = SentAnaWorker(train, test, embeddings)
#    worker2.setClassifier("randomforest", forest_args)
#    worker2.start()

if __name__ == '__main__':
    main()
    
    
"""
binary_vectorizer = CountVectorizer(binary=True) # try False - use term frequency rates
X_train_v = binary_vectorizer.fit_transform(X_train)
print X_train_v.shape
print len(binary_vectorizer.vocabulary_)
X_test_v = binary_vectorizer.transform(X_test)
X_test_v.shape
classifier = LogisticRegression()
classifier.fit(X_train_v, y_train)
# predict the y for the test data
predictions = classifier.predict(X_test_v)

for i in range(10):
    print y_test[i], predictions[i]
    print 'accuracy', classifier.score(X_test_v, y_test)
print 'recall', recall_score(y_test, predictions)
print 'precision', precision_score(y_test, predictions)
print 'f1', f1_score(y_test, predictions)

from sklearn.grid_search import GridSearchCV

pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('clf', LogisticRegression())
])

parameters = {
    'vect__binary': (True, False),
}

grid_search = GridSearchCV(pipeline, parameters, n_jobs=1, verbose=3, scoring='f1')
grid_search.fit(X_train, y_train)

print 'Best score:', grid_search.best_score_
print 'Best parameters set:'
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print '\t', param_name, best_parameters[param_name]
    
    test = pd.read_table("test.tsv")
X_actual_test = test['Phrase']
X_actual_test = binary_vectorizer.transform(X_actual_test)
X_actual_test.shape
# predict the y for the test data
actual_predictions = classifier.predict(X_actual_test)

for i in range(10):
    print y_test[i], predictions[i]
    
from sklearn.ensemble import RandomForestClassifier

pipeline = Pipeline([
    ('clf', RandomForestClassifier(criterion='entropy'))
])

parameters = {
    'clf__n_estimators': (5, 10, 20, 50),
    'clf__max_depth': (50, 150, 250),
    'clf__min_samples_split': (1, 2, 3),
    'clf__min_samples_leaf': (1, 2, 3)
}

grid_search = GridSearchCV(pipeline, parameters, n_jobs=1, verbose=3, scoring='f1')
grid_search.fit(X_train, y_train)

print 'Best score:', grid_search.best_score_
print 'Best parameters set:'
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print '\t', param_name, best_parameters[param_name]
    
{"classifier":"randomforest",
 "classifier_args":{"n_estimators": 100, "min_samples_leaf":10, "n_jobs":-1},
 "lowercase":"true",
 "map_to_synsets":"true",
 "map_to_lex":"true",
 "duplicates":"true"
}
"""