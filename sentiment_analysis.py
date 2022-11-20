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

    cleaned = []

    for entry in text:

        #gets rid of any urls in the text
        entry = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', entry)
        #removes punctuation from entry
        nopunc = [char for char in entry if char not in string.punctuation]
        #removes the numbers from entry
        nopunc = ''.join([i for i in nopunc if not i.isdigit()])
        #removes the stopwords from entry
        nopunc =  [word.lower() for word in nopunc.split() if word not in stopwords.words('english')]
        #removes all non english words from entry
        nopunc = [word for word in nopunc if word in words]
        #removes all words that were in the search  query
        nopunc = [word for word in nopunc if word != 'this']
        #returns the stemmed entry
        nopunc = [stemmer.lemmatize(word) for word in nopunc]

        cleaned.append(' '.join(nopunc))

    return cleaned


## should take in a list of strings containing all the vectors. These need to be cleaned first. Should probably include
## stemmin etc
def tf_idf(processed_text):

    vectoriser = TfidfVectorizer()
    X = vectoriser.fit_transform(processed_text)

    return vectoriser


def test_processing(fileName):
    descriptions = []
    prices = []
    address = []
    processed_text = []
    labels = []

    if(fileName==''):
        print('No pre-processed file given.')
        with open('CleanedScraperOutput.csv','r') as csvfile:
            c = csv.reader(csvfile, delimiter = ',')
            next(csvfile)
            for row in c:
                processed_text.append(str(row[8]))
                prices.append(int(row[7]))
                address.append(str(row[2]))
        processed_text = text_process(processed_text)
        labels = address_labels(address)
        print('finished text processing and labelling')

        ## ensures there are at least 2 entries for each dublin area
        size_labels = [len([value for value in labels if value == area]) for area in range(-1,25)]
        for idx, num in enumerate(size_labels):
            if(num<2):
                for i, label in enumerate(labels):
                    if(label == idx - 1):
                        labels.pop(i)
                        processed_text.pop(i)
    else:
        print('Reading from pre-processed file: ', fileName)
        with open(fileName,'r') as csvfile:
            c = csv.reader(csvfile, delimiter = ',')
            next(csvfile)
            for row in c:
                processed_text.append(row[1])
                labels.append(int(row[2]))


    X_train, X_test, y_train, y_test = train_test_split(processed_text, labels, test_size=0.2, stratify=labels)

    if(fileName == ''):
        dict = {'Description': processed_text, 'AddressLabel': labels}
        df = pd.DataFrame(dict)
        df.to_csv('ProcessedText.csv')


    tfidf_vec = TfidfVectorizer(min_df = 1, token_pattern = r'[a-zA-Z]+')
    X_train_bow = tfidf_vec.fit_transform(X_train) # fit train
    X_test_bow = tfidf_vec.transform(X_test) # transform test
    model_svm = svm.SVC(C=8.0, kernel='linear')
    model_svm.fit(X_train_bow, y_train)
    predictions = model_svm.predict(X_test_bow)

    print(model_svm.score(X_test_bow, y_test))



##if you have new data and need to pre-process
test_processing('')

## if you have already performed the pre-processing
test_processing('ProcessedText.csv')
