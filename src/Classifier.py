from sklearn.feature_extraction import text
from sklearn.feature_selection import chi2

from nltk.stem import WordNetLemmatizer #for ignoring common words

from src.HTML_Extractor import *
import pandas as pd
import numpy as np
import os
import csv


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


def locate_file(dir, year, cik):
    folders = [folder for folder in os.listdir(dir) if re.match(year, folder)]
    for folder in folders:
        for form in os.listdir(dir+folder):
            if form.split('-')[0] == cik:
                return dir+folder+'/'+form
    return ''


path_to_csv = 'src/label_reference.csv'
df_csv = pd.read_csv(open(path_to_csv, 'rb'))

directory = '/mnt/volume/10-K/10-K_files/'
output_folder = '/mnt/volume/10-K/10-K_files/'

my_stop_words = text.ENGLISH_STOP_WORDS
lemmatizer = WordNetLemmatizer()


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
# convert_html(directory+'False_Set/', directory+'False_Set_Processed/')
# convert_html(directory+'Truth_Set/', directory+'Truth_Set_Processed/')

#making csv from processed false and truth sets
# csv_out = open('processed_10-K.csv', mode='w')
# writer = csv.writer(csv_out)
# fields = ['cik', 'company name', 'date', 'full text', 'prosecution']
# paths = ['False_Set_Processed/', 'Truth_Set_Processed/']
# for path in paths:
#     for filename in os.listdir(directory+path):
#         page = open(directory+path+filename)
#         soup = bs(page.read(), "lxml")
#         filename = filename.split('-')
#         date = filename[4] + '-' + filename[5] + '-' + filename[6][0:2]
#         label = 0 if path == paths[0] else 1
#         writer.writerow([filename[0], filename[1], date, soup.text, label]);
#
# csv_out.close()
print('new files found: ', counter)


#lemmatize(form_text_T)
#lemmatized_truth_forms = tfidf.fit_transform(form_text_T);
#feature_names = tfidf.get_feature_names()
#print(topTerms(tfidf, feature_names)







