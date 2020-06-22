from sklearn.feature_extraction import text
from sklearn.feature_selection import chi2
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

from nltk.stem import WordNetLemmatizer #for ignoring common words
# from nltk.tokenize import PunktSentenceTokenizer, ToktokTokenizer
from src.HTML_Extractor import *
import pandas as pd
import numpy as np
import os
import csv
import sys
import matplotlib.pylab as plt
import shutil

en_stop = text.ENGLISH_STOP_WORDS
stemmer = WordNetLemmatizer()

def process_text(article):
    # remove all mentions of the title
    article = re.sub(r'((ITEM)\s*8)|FINANCIAL\s*STATEMENTS\s*AND\s*SUPPLEMENTARY?\s*DATA', '', article.lower())
    # Remove all the special characters
    article = re.sub(r'\W', ' ', str(article))

    # remove all single characters
    article = re.sub(r'\s+[a-zA-Z]\s+', ' ', article)

    # Remove single characters from the start
    article = re.sub(r'\^[a-zA-Z]\s+', ' ', article)

    # Substituting multiple spaces with single space
    article = re.sub(r'\s+', ' ', article, flags=re.I)

    # Remove all stand alone digits
    article = re.sub(r'\s*[\d+]', ' ', article)

    # Removing prefixed 'b'
    article = re.sub(r'^b\s+', '', article)

    # Converting to Lowercase
    article = article.lower()

    # Lemmatization
    tokens = article.split()
    tokens = [stemmer.lemmatize(word) for word in tokens]
    tokens = [word for word in tokens if word not in en_stop]

    article = ' '.join(tokens)
    return article


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


def locate_file(directory, regex, cik):
    folders = [folder for folder in os.listdir(directory) if re.match(regex, folder)]
    for folder in folders:
        for form in os.listdir(directory+folder):
            if form.split('-')[0] == cik:
                return directory+folder+'/'+form
    return ''


def prior_years(year, num_prior):
    years = []
    try:
        num_year = int(year)
        for index in range(num_prior):
            years.append(num_year-index-1)
            print('{0} was added'.format(num_year-index-1))
            print(years)
    except ValueError:
        print("not a valid number")

    return [str(y) for y in years]


def locate_prior_files(directory, year, quarter, cik, num_prior):
    years = prior_years(year, num_prior)
    folders = [(yr+quarter)for yr in years]
    print(folders)
    prior_files = []
    for folder in folders:
        print('directory {0} and folder {1}'.format(directory, folder))
        prior_file = locate_file(directory, folder, cik)
        if prior_file != '':
            prior_files.append(prior_file)
        else:
            print('prior file could not be found in {0}'.format(folder))
    return prior_files


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
            writer.writerow([filename[0], filename[1], date, process_text(soup.text), label]);
    csv_out.close()


def top_termsNB(X_raw, y, vectorizer):
    X = vectorizer.fit_transform(X_raw)
    feature_names = vectorizer.get_feature_names()
    clf = MultinomialNB(alpha=0.2, fit_prior=True)
    clf.fit(X, y)
    print('naive bayes shape: ',clf.feature_count_.shape)
    non_pros_token_count = clf.feature_count_[0, :] + 1
    pros_token_count = clf.feature_count_[1, :] + 1
    tokens = pd.DataFrame({'token': feature_names, 'non_pros': non_pros_token_count, 'pros': pros_token_count}).set_index(
        'token')
    tokens['non_pros'] = tokens.non_pros / clf.class_count_[0]
    tokens['pros'] = tokens.pros / clf.class_count_[1]
    tokens['pros_ratio'] = tokens.pros / tokens.non_pros
    print('top 10 terms using new method')
    print(tokens.sort_values('pros_ratio', ascending=False)[:10])

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
    print(list(feature_names[x] for x in top_positive_coefficients))
    # top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
    # # create plot
    # plt.figure(figsize=(15, 5))
    # colors = [‘red’ if c < 0 else ‘blue’ for c in coef[top_coefficients]]
    # plt.bar(np.arange(2 * top_features), coef[top_coefficients], color=colors)
    # feature_names = np.array(feature_names)
    # plt.xticks(np.arange(1, 1 + 2 * top_features), feature_names[top_coefficients], rotation=60, ha=’right’)
    # plt.show()


def cross_validation_cm(pipeline, params, X_train, X_test, y_train, y_test):
    clf = GridSearchCV(pipeline, params, cv=5)
    clf.fit(X_train, y_train)
    print('cross validation scores for {} {}'.format(pipeline.steps[1][1].__class__.__name__, pipeline.steps[0][1].__class__.__name__))
    print('Best Score: ', clf.best_score_)

    print('Best Params: ', clf.best_params_)

    # generate confusion matrix
    y_pred = clf.best_estimator_.predict(X_test)
    print('Prediction Accuracy: ', accuracy_score(y_test, y_pred))
    null_accuracy = y_test.value_counts().head(1) / len(y_test)
    print('Null Accuracy: ', null_accuracy)
    plot_confusion_matrix(y_test, y_pred,
                          classes=['NonProsecution', 'Prosecution'],
                          title='Confusion matrix, without normalization')
    plt.savefig('unnormalized graph {} {}.png'.format(pipeline.steps[1][1].__class__.__name__, pipeline.steps[0][1].__class__.__name__))
    plot_confusion_matrix(y_test, y_pred,
                          classes=['NonProsecution', 'Prosecution'], normalize=True,
                          title='Normalized confusion matrix')
    plt.savefig('normalized graph {} {}.png'.format(pipeline.steps[1][1].__class__.__name__, pipeline.steps[0][1].__class__.__name__))


def classify_unlabeled_set(pipeline, params, X_train, y_train, X_pred):
    clf = GridSearchCV(pipeline, params, cv=5)
    clf.fit(X_train, y_train)
    print('cross validation scores for {} {}'.format(pipeline.steps[1][1].__class__.__name__,
                                                     pipeline.steps[0][1].__class__.__name__))
    print('Best Score: ', clf.best_score_)

    print('Best Params: ', clf.best_params_)
    y_pred = clf.best_estimator_.predict(X_pred)
    return y_pred;


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

path_to_csv = 'label_reference.csv' if sys.platform == 'darwin' else 'src/label_reference.csv'
df_csv = pd.read_csv(open(path_to_csv, 'rb'))

directory = '/Users/Ju1y/Documents/GIES Research Project/10-K/' if sys.platform == 'darwin' else '/mnt/volume/10-K/10-K_files/'

tfidf = TfidfVectorizer(ngram_range=(1,2),
                                stop_words=en_stop,
                                min_df=2,
                                sublinear_tf=True,
                                norm=None,
                                binary=True)
countv = CountVectorizer(ngram_range=(1,2), stop_words=en_stop, min_df=2, max_df=.5)
# Scans label sheet and locates corresponding 10-K forms
# counter = 0
# for ind in df_csv.index:
#     cik = df_csv['cik'][ind]
#     date = df_csv['datadate'][ind]
#     if not pd.isnull(date) and not pd.isnull(cik):
#         date.astype(np.int64)
#         cik = int(cik)
#         month_date = str(int(date))[4:]
#         if month_date == '1231':
#             actual_year = int(str(date)[:4]) + 1
#             file = locate_file(directory, str(actual_year), str(cik))
#             if file != '':
#                 dispo = df_csv['disposition'][ind]
#                 head, tail = os.path.split(file)
#                 if dispo == 1:
#                     os.rename(file, directory+'False_Set/'+tail)
#                 else:
#                     os.rename(file, directory+'Truth_Set/'+tail)
num_years_prior = 3  # how many years prior would the program examine
for ind in df_csv.index:
    cik = df_csv['cik'][ind]
    date = df_csv['datadate'][ind]
    dispo = df_csv['disposition'][ind]
    if not pd.isnull(date) and not pd.isnull(cik):
        date.astype(np.int64)
        cik = int(cik)
        month_date = str(date)[4:]
        actual_year = (int(str(date)[:4]) + 1) if month_date == '1231' else str(date)[:4]
        file = locate_file(directory, str(actual_year), str(cik))
        quarter = ''
        if file == '':
            if dispo == 1:
                file = locate_file(directory, 'Truth_Set', str(cik))
            else:
                file = locate_file(directory, 'False_Set', str(cik))
        else:
            quarter = file.split('/')[-2][4:]
            print('Quarter found to be {0}'.format(quarter))
        if file != '':
            head, tail = os.path.split(file)
            print('file {0} successfully found'.format(tail))
            if not pd.isnull(df_csv['settle'][ind]) and int(df_csv['settle'][ind]) != 0:
                shutil.copyfile(file, directory + 'Disclosure/' + tail)
            else:
                shutil.copyfile(file, directory + 'Non_Disclosure/' + tail)
            before = locate_prior_files(directory, str(actual_year), quarter, str(cik), num_years_prior)
            for index_prior in range(len(before)):
                head, tail = os.path.split(before[index_prior])
                if int(df_csv['settle_m{0}'.format(index_prior+1)][ind]) != '0':
                    shutil.copyfile(before[index_prior], directory+'Disclosure/'+tail)
                else:
                    shutil.copyfile(before[index_prior], directory + 'Non_Disclosure/' + tail)





        # Processes 10-K forms in the truth and false set folders
# convert_html(directory+'False_Set/', directory+'False_Set_Processed/')
# convert_html(directory+'Truth_Set/', directory+'Truth_Set_Processed/')
#
# # outputting processed false and truth sets into csv
# output_csv('processed_10-K.csv',  # csv name
#            ['cik', 'company name', 'date', 'full text', 'prosecution'],  # column names
#            ['False_Set_Processed/', 'Truth_Set_Processed/'],  # folder names in which the files are extracted from
#            directory)  # directory in which the files are extracted from
#
# # making csv from processed false and truth sets
# df_all_forms = pd.read_csv('processed_10-K.csv', usecols=['full text', 'prosecution'])
# df_all_forms['full text'] = df_all_forms['full text'].values.astype('U')
# df_all_forms['prosecution'] = df_all_forms['prosecution'].values.astype('U')
#
# print('new files found: ', counter)
# # Splitting dataset for classification
# y = [int(pros) for pros in df_all_forms['prosecution']]
#
# # performing Naive Bayes test
# print('MultinomialNB analysis with tfidf...\n')
# top_termsNB(df_all_forms['full text'], y, tfidf)
# print('-'*20, '\n')
# print('MultinomialNB analysis with countvectorizer...\n')
# top_termsNB(df_all_forms['full text'], y, countv)
# print('-'*20, '\n')
# print('Linear SVM analysis with tfidf...\n')
# X = tfidf.fit_transform(df_all_forms['full text'])
# feature_names = tfidf.get_feature_names()
# svm = LinearSVC(C=0.01, dual=False, max_iter=1000, penalty='l2')
# svm.fit(X, y)
# top_terms(svm, feature_names)
# print('-'*20, '\n')
# print('Linear SVM analysis with tfidf...\n')
# X = countv.fit_transform(df_all_forms['full text'])
# feature_names = countv.get_feature_names()
# svm = LinearSVC(C=0.01, dual=False, max_iter=1000, penalty='l2')
# svm.fit(X, y)
# top_terms(svm, feature_names)
# print('-'*20, '\n')
# print('Chi2 analysis with tfidf...\n')
#
# # performing chi2 test
# chi2_analysis(tfidf, df_all_forms, 20)
# print('Chi2 analysis with countvectorizer...\n')
# chi2_analysis(countv, df_all_forms, 20)


mnb_pipeline = Pipeline([
    ('tfidf_pipeline', TfidfVectorizer()),
    ('mnb', MultinomialNB())
])
mnbcount_pipeline = Pipeline([
    ('countvec', CountVectorizer()),
    ('mnb', MultinomialNB())
])
svm_pipeline = Pipeline([
    ('tfidf_pipeline', TfidfVectorizer()),
    ('linearsvm', LinearSVC())
])
svmcount_pipeline = Pipeline([
    ('countvec', CountVectorizer()),
    ('linearsvm', LinearSVC())
])
# different parameter settings to test out
mnb_params = {
    'mnb__alpha': [.2],
    'mnb__fit_prior': [True],
    'tfidf_pipeline__ngram_range': [(1,2)],
    'tfidf_pipeline__max_df': [.5, .7, 1.0],
    'tfidf_pipeline__min_df': [2],
    'tfidf_pipeline__binary': [True],
    'tfidf_pipeline__norm': [None],
}
mnbcount_params = {
    'mnb__alpha': [.3],
    'mnb__fit_prior': [True],
    'countvec__ngram_range': [(1,2)],
    'countvec__max_df': [.5],
    'countvec__min_df': [2]
}
svm_params = {
    'linearsvm__C': np.arange(0.01, 100, 10),
    'linearsvm__penalty': ['l2'],
    'linearsvm__dual': [False],
    'linearsvm__max_iter': [1000],
    'tfidf_pipeline__ngram_range': [(1,2)],
    'tfidf_pipeline__min_df': [2],
    'tfidf_pipeline__binary': [True],
    'tfidf_pipeline__norm': [None],
}
svmcount_params = {
    'countvec__ngram_range': [(1, 2)],
    'countvec__max_df': [.5],
    'countvec__min_df': [2],
    'linearsvm__C': np.arange(0.01, 100, 10),
    'linearsvm__penalty': ['l2'],
    'linearsvm__dual': [False],
    'linearsvm__max_iter': [1000],
}

#classifying rest of the unlabeled set
# for folder in [folder for folder in os.listdir(directory) if re.match(r'\d+', folder)]:
#     convert_html(directory + folder + "/", directory + 'Unlabeled_Set_Processed/')
#
# output_csv('unlabeled_10-K.csv',  # csv name
#            ['cik', 'company name', 'date', 'full text', 'label'],  # column names
#            ['Unlabeled_Set_Processed'],  # folder names in which the files are extracted from
#            directory)
# df_unlabeled_forms = pd.read_csv('unlabeled_10-K.csv', usecols=['full text', 'label'])
# df_unlabeled_forms['full text'] = df_all_forms['full text'].values.astype('U')
# df_unlabeled_forms['label'] = df_all_forms['label'].values.astype('U')
#
#
# df_unlabeled_forms['label'] = classify_unlabeled_set(mnb_pipeline,
#                                                      mnb_params,
#                                                      df_all_forms['full text'],
#                                                      df_all_forms['prosecution'],
#                                                      df_unlabeled_forms['full text'])

# Prints confusion matrix for different classifiers
# full_text_train, full_text_test, label_train, label_test = train_test_split(df_all_forms['full text'],
#                                                                             df_all_forms['prosecution'],
#                                                                             test_size=0.2, random_state=85)
# # print('mnb with tfidf')
# print('-'*20)
# cross_validation_cm(mnb_pipeline, mnb_params, full_text_train, full_text_test, label_train, label_test)
# print('mnb with countvec')
# print('-'*20)
# cross_validation_cm(mnbcount_pipeline, mnbcount_params, full_text_train, full_text_test, label_train, label_test)
# print('svm with tfidf')
# print('-'*20)
# cross_validation_cm(svm_pipeline, svm_params, full_text_train, full_text_test, label_train, label_test)
# print('svm with count vectorizer')
# print('-'*20)
# cross_validation_cm(svmcount_pipeline, svmcount_params, full_text_train, full_text_test, label_train, label_test)



# NB_optimal = MultinomialNB(alpha=.1, fit_prior=True)
# X_train = tfidf.fit_transform(df_all_forms['full text'])
# y_train = [int(pros) for pros in df_all_forms['prosecution']]
# NB_optimal.fit(X_train, y_train)
# pos_class_prob_sorted = NB_optimal.feature_log_prob_[1, :].argsort()
# print('Most associative words ------')
# print(np.take(tfidf.get_feature_names(), pos_class_prob_sorted[:10]))

#feature_names = tfidf.get_feature_names()
#print(topTerms(tfidf, feature_names)







