from sklearn.feature_extraction import text
from sklearn.feature_selection import chi2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from nltk.stem import WordNetLemmatizer #for ignoring common words
import nltk
nltk.download('wordnet')
from src.HTML_Extractor import *
import pandas as pd
import numpy as np
import os
import csv


def lemmatize(dataset, stop_words):
    for article in dataset:
        article = ' '.join([lemmatizer.lemmatize(i)
                         for i in article.split() if i not in stop_words])
    return dataset


# This calculates the idf value for the terms in the posts and prints the highest ones
def topTermsIDF(vectorizer):
    features = vectorizer.get_feature_names()
    indices = np.argsort(vectorizer.idf_)[::-1]
    top_n = 40
    top_features = dict(zip([features[i] for i in indices[:]], [vectorizer.idf_[i] for i in indices[:]]))
    return top_features


def chi2_analysis(vectorizer, df_form, n_terms, lemmatization):
    if lemmatization:
        df_form['full text'] = lemmatize(df_form['full text'], text.ENGLISH_STOP_WORDS)
    response_all = vectorizer.fit_transform(df_form['full text'])
    set_array = response_all.toarray()
    features_chi2 = chi2(set_array, df_form['prosecution'])
    indices = np.argsort(features_chi2[0])
    feature_names = np.array(vectorizer.get_feature_names())[indices]

    unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
    bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
    print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-n_terms:])))
    print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-n_terms:])))

    feature_names = feature_names[::-1]

    print(feature_names[:n_terms])


def locate_file(dir, year, cik):
    folders = [folder for folder in os.listdir(dir) if re.match(year, folder)]
    for folder in folders:
        for form in os.listdir(dir+folder):
            if form.split('-')[0] == cik:
                return dir+folder+'/'+form
    return ''


def topTermsNB(df_form, vectorizer):
    X = vectorizer.fit_transform(df_form['full text'])
    words = vectorizer.get_feature_names()
    y = [int(pros) for pros in df_form['prosecution']]

    clf = MultinomialNB(alpha=0)
    clf.fit(X, y)
    likelihood_df = pd.DataFrame(clf.feature_log_prob_.transpose(),
                                 columns=['No_Prosecution', 'Prosecution'],
                                 index=words)
    likelihood_df['Relative Prevalence for Prosecution'] = likelihood_df.eval('(exp(Prosecution) - exp(No_Prosecution))')
    print('Top 10 terms strongly associated to Prosecution according to Naive Bayes analysis:\n')
    print(likelihood_df['Relative Prevalence for Prosecution'].sort_values(ascending=False).iloc[:10])
    return likelihood_df


path_to_csv = 'src/label_reference.csv'  # 'label_reference.csv' #
df_csv = pd.read_csv(open(path_to_csv, 'rb'))

directory = '/mnt/volume/10-K/10-K_files/'  # '/Users/Ju1y/Documents/GIES Research Project/10-K/' #

my_stop_words = text.ENGLISH_STOP_WORDS
lemmatizer = WordNetLemmatizer()
tfidf = tfidf = TfidfVectorizer(ngram_range=(1,2), stop_words=my_stop_words, min_df=2, sublinear_tf=True)
form_text_T = []
form_text_F = []


# Scans label sheet and locates corresponding 10-K forms
counter = 0
for ind in df_csv.index:
    cik = df_csv['cik'][ind]
    date = df_csv['datadate'][ind]
    if not pd.isnull(date) and not pd.isnull(cik):
        date.astype(np.int64)
        cik = int(cik)
        month_date = str(int(date))[4:]
        if month_date == '1231':
            actual_year = int(str(date)[:4]) + 1
            file = locate_file(directory, str(actual_year), str(cik))
            if file != '':
                dispo = df_csv['disposition'][ind]
                head, tail = os.path.split(file)
                if dispo == 1:
                    os.rename(file, directory+'False_Set/'+tail)
                else:
                    os.rename(file, directory+'Truth_Set/'+tail)

# Processes 10-K forms in the truth and false set folders
convert_html(directory+'False_Set/', directory+'False_Set_Processed/')
convert_html(directory+'Truth_Set/', directory+'Truth_Set_Processed/')

#making csv from processed false and truth sets
csv_out = open('processed_10-K.csv', mode='w')
writer = csv.writer(csv_out)
fields = ['cik', 'company name', 'date', 'full text', 'prosecution']
paths = ['False_Set_Processed/', 'Truth_Set_Processed/']
writer.writerow(fields)
for path in paths:
    for filename in os.listdir(directory+path):
        page = open(directory+path+filename)
        soup = bs(page.read(), "lxml")
        filename = filename.split('-')
        date = filename[4] + '-' + filename[5] + '-' + filename[6][0:2]
        label = 0 if path == paths[0] else 1
        if label == 0:
            form_text_F.append(soup.text)
        else:
            form_text_T.append(soup.text)
        writer.writerow([filename[0], filename[1], date, soup.text, label]);
df_all_forms = pd.read_csv('processed_10-K.csv', usecols=['full text', 'prosecution'])
csv_out.close()
print('new files found: ', counter)

lemmatized_truth_forms = lemmatize(form_text_T, my_stop_words);
lemmatized_false_forms = lemmatize(form_text_F, my_stop_words);
topTermsNB(df_all_forms, tfidf)
print('-'*20, '\n')
print('Chi2 analysis...\n')
chi2_analysis(tfidf, df_all_forms, 20, True)
#feature_names = tfidf.get_feature_names()
#print(topTerms(tfidf, feature_names)







