from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from sklearn.feature_selection import chi2

from sklearn.naive_bayes import MultinomialNB, BernoulliNB

from nltk.stem import WordNetLemmatizer #for ignoring common words

from src.HTML_Extractor import *
import pandas as pd
import numpy as np
import os


def lemmatize(dataset):
    for article in dataset:
        article = ' '.join([lemmatizer.lemmatize(i)
                         for i in text.split() if i not in my_stop_words])
    return dataset


# This calculates the idf value for the terms in the posts and prints the highest ones
def topTerms(vectorizer, features):
    indices = np.argsort(vectorizer.idf_)[::-1]
    top_n = 40
    top_features = dict(zip([features[i] for i in indices[:]], [vectorizer.idf_[i] for i in indices[:]]))
    return top_features


def chi2_analysis(vectorizer, dataset, n_terms, lemmatization):
    if lemmatization:
        lemmatize(dataset)
    response_all = vectorizer.fit_transform(dataset)
    set_array = response_all.toarray()
    features_chi2 = chi2(set_array)

path_to_csv = '/Users/Ju1y/Documents/GIES Research Project/10-K form excel/label_reference.csv'
df_csv = pd.read_csv(open(path_to_csv, 'rb'))

directory = '/Users/Ju1y/Documents/GIES Research Project/10-K/2010Q1/'
output_folder = '/Users/Ju1y/Documents/GIES Research Project/Item8/'

my_stop_words = text.ENGLISH_STOP_WORDS
lemmatizer = WordNetLemmatizer()

form_text = []
form_text_T = []
form_text_F = []

#Data Import and split into truth and false sets

for filename in os.listdir(directory):
    if filename.endswith('.html'):
        cik = filename.split('-')[0];
        companies_found = df_csv.loc[df_csv['cik'] == float(cik)];
        if not companies_found.empty:
            print(companies_found)





#lemmatize(form_text_T)
#lemmatized_truth_forms = tfidf.fit_transform(form_text_T);
#feature_names = tfidf.get_feature_names()
#print(topTerms(tfidf, feature_names)







