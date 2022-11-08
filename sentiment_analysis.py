import pandas as pd
import csv
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
import string
import re
from collections import Counter
import random
from sklearn import svm



# takes in the price per month and returns a score between 0 and 1
def price_labels(prices):

    labels = []

    for price in prices:
        labels.append(random.randint(0,1))

        """
        if(int(price) <1000):
            labels.append(0)
        elif(int(price) < 2000):   #1000 < price < 2000
            labels.append(1)
        elif(int(price) <2500):    #2000 < price < 2500
            labels.append(2)
        elif(int(price) < 3000):   #2500 < price < 3000
            labels.append(3)
        elif(int(price) < 3500):   #3000 < price < 3500
            labels.append(4)
        elif(int(price) < 5000):   #3500 < price < 5000
            labels.append(5)
        else:                  # 5000 < price
            labels.append(6)

        """


    return labels


def text_process(text):
    """
    Performs text processing on the english language. Removes punctuation, urls, numbers,
    non-english words, the search query and stopwords from the text. Then performs lemmatising.

    Parameters:
        text (str): the text to process

    Returns:
        nopunc ([str]): the processed text

    """

    #sets the english words and the stemmer being used
    words = set(nltk.corpus.words.words())
    stemmer = WordNetLemmatizer()

    #gets rid of any urls in the text
    text = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', text)
    #removes punctuation from text
    nopunc = [char for char in text if char not in string.punctuation]
    #removes the numbers from text
    nopunc = ''.join([i for i in nopunc if not i.isdigit()])
    #removes the stopwords from text
    nopunc =  [word.lower() for word in nopunc.split() if word not in stopwords.words('english')]
    #removes all non english words from text
    nopunc = [word for word in nopunc if word in words]
    #removes all words that were in the search  query
    nopunc = [word for word in nopunc if word != 'this']
    #returns the stemmed text
    nopunc = [stemmer.lemmatize(word) for word in nopunc]

    return nopunc


## should take in a list of strings containing all the vectors. These need to be cleaned first. Should probably include
## stemmin etc
def tf_idf(processed_text):

    vectoriser = TfidfVectorizer()
    X = vectoriser.fit_transform(processed_text)

    return vectoriser


def test_processing():
    descriptions = []
    prices = []
    processed_text = []
    with open('data_test1.csv','r') as csvfile:
        c = csv.reader(csvfile, delimiter = ',')

        next(csvfile)
        for row in c:
            if str(row[6])!= '':
                processed_text.append(' '.join(text_process(str(row[6]))))
                prices.append(str(row[5]))
    labels = price_labels(prices)

    documents = [(processed_text[idx], labels[idx]) for idx, text in enumerate(processed_text)]



    train, test = train_test_split(documents, test_size = 0.2, random_state=42)


    X_train = [words for (words, label) in train]
    X_test = [words for (words, label) in test]
    y_train = [label for (words, label) in train]
    y_test = [label for (words, label) in test]


    tfidf_vec = TfidfVectorizer(min_df = 1, token_pattern = r'[a-zA-Z]+')
    X_train_bow = tfidf_vec.fit_transform(X_train) # fit train
    X_test_bow = tfidf_vec.transform(X_test) # transform test



    model_svm = svm.SVC(C=8.0, kernel='linear')
    model_svm.fit(X_train_bow, y_train)

    predictions = model_svm.predict(X_test_bow)
