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
import matplotlib.pyplot as plt
import random
from sklearn import svm



# takes in the Dublin postcode and returns the score
def address_labels(address):

    labels = []

    for label in address:
        if(label == 'Other' or label == ''):
            labels.append(0)
        elif(label =='6w' or label == '6W'):
            labels.append(-1)
        else:
            labels.append(int(label))


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
    address = []
    processed_text = []
    with open('CleanedScraperOutput.csv','r') as csvfile:
        c = csv.reader(csvfile, delimiter = ',')

        next(csvfile)
        for row in c:
            processed_text.append(' '.join(text_process(str(row[6]))))
            address.append(str(row[2]))
    print(len(processed_text))
    labels = address_labels(address)

    plt.hist(labels, 25)
    plt.xlabel('Post Code (0=Dublin, -1=6W)')
    plt.title('Distribution of Dublin Postcode data')
    plt.xticks([num for num in range(-1,25)], [str(num) for num in range(-1,25)])
    plt.show()

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
    print(y_test)
    print(list(predictions))
    print('Accuracy: ', 100*len([value for idx, value in enumerate(predictions) if value == y_test[idx]])/len(y_test))


test_processing()
