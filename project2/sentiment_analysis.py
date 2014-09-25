import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cross_validation import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
import re


def main():
    train_file = open(sys.argv[1])
    test_file = open(sys.argv[2])
    brown_file = open(sys.argv[3])

if __name__ == '__main__':
    main()
    
