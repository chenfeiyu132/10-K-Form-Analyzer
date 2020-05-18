from sklearn.feature_extraction import text
from sklearn.feature_selection import chi2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC

from nltk.stem import WordNetLemmatizer #for ignoring common words
from nltk.tokenize import PunktSentenceTokenizer, ToktokTokenizer
from src.HTML_Extractor import *
import pandas as pd
import numpy as np
import os
import csv
import sys
from matplotlib import pyplot as plt

en_stop = text.ENGLISH_STOP_WORDS
stemmer = WordNetLemmatizer()

def process_text(dataset):
    for index in range(len(dataset)):
        # remove all mentions of the title
        dataset[index] = re.sub(r'((ITEM)\s*8)|FINANCIAL\s*STATEMENTS\s*AND\s*SUPPLEMENTARY?\s*DATA', '', dataset[index].lower())
        # Remove all the special characters
        dataset[index] = re.sub(r'\W', ' ', str(dataset[index]))

        # remove all single characters
        dataset[index] = re.sub(r'\s+[a-zA-Z]\s+', ' ', dataset[index])

        # Remove single characters from the start
        dataset[index] = re.sub(r'\^[a-zA-Z]\s+', ' ', dataset[index])

        # Substituting multiple spaces with single space
        dataset[index] = re.sub(r'\s+', ' ', dataset[index], flags=re.I)

        # Remove all stand alone digits
        dataset[index] = re.sub(r'\s*[\d+]', ' ', dataset[index])

        # Removing prefixed 'b'
        dataset[index] = re.sub(r'^b\s+', '', dataset[index])

        # Converting to Lowercase
        dataset[index] = dataset[index].lower()

        # Lemmatization
        tokens = dataset[index].split()
        tokens = [stemmer.lemmatize(word) for word in tokens]
        tokens = [word for word in tokens if word not in en_stop]

        dataset[index] = ' '.join(tokens)
    return dataset


# This calculates the idf value for the terms in the posts and prints the highest ones
def topTermsIDF(vectorizer):
    features = vectorizer.get_feature_names()
    indices = np.argsort(vectorizer.idf_)[::-1]
    top_n = 40
    top_features = dict(zip([features[i] for i in indices[:]], [vectorizer.idf_[i] for i in indices[:]]))
    return top_features


def chi2_analysis(vectorizer, df_form, n_terms):
    response_all = vectorizer.fit_transform(df_form['full text'])
    set_array = response_all.toarray()
    features_chi2 = chi2(set_array, df_form['prosecution'] == '1')
    indices = np.argsort(features_chi2[0])
    feature_names = np.array(vectorizer.get_feature_names())[indices]
    feature_names = feature_names[::-1]
    unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
    bigrams = [v for v in feature_names if len(v.split(' ')) == 2]

    print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[:n_terms])))
    print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[:n_terms])))
    print(feature_names[:n_terms])


def locate_file(dir, year, cik):
    folders = [folder for folder in os.listdir(dir) if re.match(year, folder)]
    for folder in folders:
        for form in os.listdir(dir+folder):
            if form.split('-')[0] == cik:
                return dir+folder+'/'+form
    return ''


def output_csv(filename, fields, paths, directory_name):
    csv_out = open(filename, mode='w')
    writer = csv.writer(csv_out)
    writer.writerow(fields)
    for path in paths:
        for filename in os.listdir(directory_name + path):
            page = open(directory_name + path + filename)
            soup = bs(page.read(), "lxml")
            filename = filename.split('-')
            date = filename[4] + '-' + filename[5] + '-' + filename[6][0:2]
            label = 0 if path == paths[0] else 1
            writer.writerow([filename[0], filename[1], date, soup.text, label]);
    csv_out.close()


def top_termsNB(X, y, feature_names):
    clf = MultinomialNB(alpha=0.1, fit_prior=True)
    clf.fit(X, y)
    likelihood_df = pd.DataFrame(clf.feature_log_prob_.transpose(),
                                 columns=['No_Prosecution', 'Prosecution'],
                                 index=feature_names)
    likelihood_df['Relative Prevalence for Prosecution'] = likelihood_df.eval('(exp(Prosecution) - exp(No_Prosecution))')
    print('Top 10 terms strongly associated to Prosecution according to Naive Bayes analysis:\n')
    print(likelihood_df['Relative Prevalence for Prosecution'].sort_values(ascending=False)[:10])


def top_terms(classifier, feature_names, top_features=10):
    coefficients = classifier.coef_.ravel()
    top_positive_coefficients = np.argsort(coefficients)[-top_features:]
    # top_negative_coefficients = np.argsort(coef)[:top_features]
    print('Top ', top_features, ' most predictive terms for prosecution \n')
    print(top_positive_coefficients)
    # top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
    # # create plot
    # plt.figure(figsize=(15, 5))
    # colors = [‘red’ if c < 0 else ‘blue’ for c in coef[top_coefficients]]
    # plt.bar(np.arange(2 * top_features), coef[top_coefficients], color=colors)
    # feature_names = np.array(feature_names)
    # plt.xticks(np.arange(1, 1 + 2 * top_features), feature_names[top_coefficients], rotation=60, ha=’right’)
    # plt.show()


path_to_csv = 'label_reference.csv' if sys.platform == 'darwin' else 'src/label_reference.csv'
df_csv = pd.read_csv(open(path_to_csv, 'rb'))

directory = '/Users/Ju1y/Documents/GIES Research Project/10-K/' if sys.platform == 'darwin' else '/mnt/volume/10-K/10-K_files/'

tfidf = TfidfVectorizer(ngram_range=(1,2),
                                stop_words=en_stop,
                                min_df=2,
                                sublinear_tf=True,
                                norm=None,
                                binary=True)

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

# outputting processed false and truth sets into csv
output_csv('processed_10-K.csv',  # csv name
           ['cik', 'company name', 'date', 'full text', 'prosecution'],  # column names
           ['False_Set_Processed/', 'Truth_Set_Processed/'],  # folder names in which the files are extracted from
           directory)  # directory in which the files are extracted from

# making csv from processed false and truth sets
df_all_forms = pd.read_csv('processed_10-K.csv', usecols=['full text', 'prosecution'])
df_all_forms['full text'] = df_all_forms['full text'].values.astype('U')
df_all_forms['prosecution'] = df_all_forms['prosecution'].values.astype('U')

print('new files found: ', counter)
# Splitting dataset for classification
df_all_forms['full text'] = process_text(df_all_forms['full text'])
X = tfidf.fit_transform(df_all_forms['full text'])
feature_names = tfidf.get_feature_names()
y = [int(pros) for pros in df_all_forms['prosecution']]

# performing Naive Bayes test
print('MultinomialNB analysis...\n')
top_termsNB(X, y, feature_names)
print('-'*20, '\n')
print('Linear SVM analysis...\n')
svm = LinearSVC()
svm.fit(X, y)
top_terms(svm, feature_names)
print('-'*20, '\n')
print('Chi2 analysis...\n')

# performing chi2 test
chi2_analysis(tfidf, df_all_forms, 20)


mnb_pipeline = Pipeline([
    ('tfidf_pipeline', TfidfVectorizer()),
    ('mnb', MultinomialNB())
])
# different parameter settings to test out
grid_params = {
    'mnb__alpha': np.linspace(0.1, 1, 9),
    'mnb__fit_prior': [True],
    'tfidf_pipeline__ngram_range': [(1,2)],
    'tfidf_pipeline__min_df': [7],
    'tfidf_pipeline__binary': [True],
    'tfidf_pipeline__norm': [None],
}
clf = GridSearchCV(mnb_pipeline, grid_params, cv=5)
clf.fit(df_all_forms['full text'], df_all_forms['prosecution'])

print('Best Score: ', clf.best_score_)
print('Best Params: ', clf.best_params_)

# NB_optimal = MultinomialNB(alpha=.1, fit_prior=True)
# X_train = tfidf.fit_transform(df_all_forms['full text'])
# y_train = [int(pros) for pros in df_all_forms['prosecution']]
# NB_optimal.fit(X_train, y_train)
# pos_class_prob_sorted = NB_optimal.feature_log_prob_[1, :].argsort()
# print('Most associative words ------')
# print(np.take(tfidf.get_feature_names(), pos_class_prob_sorted[:10]))

#feature_names = tfidf.get_feature_names()
#print(topTerms(tfidf, feature_names)







