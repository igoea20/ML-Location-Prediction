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


## takes in the already processed training data and trains the svm. Returns the trained svm and tfidf.
def train_dataset(processed_training_data, labels):

    X_train = processed_training_data
    y_train = labels

    tfidf_vec = TfidfVectorizer(min_df = 1, token_pattern = r'[a-zA-Z]+')
    X_train_bow = tfidf_vec.fit_transform(X_train) # fit train
    model_svm = svm.SVC(C=8.0, kernel='linear')
    model_svm.fit(X_train_bow, y_train)
    #predictions = model_svm.predict(X_train_bow)

    return model_svm, tfidf_vec


#takes in the processed text and a pretrained svm and tfidf and predicts the address labels
def generate_svm_scores(testing_text, model_svm, tfidf):

    X_test_bow = tfidf.transform(testing_text) # fit train
    predictions = model_svm.predict(X_test_bow) #predictions = model_svm.predict(X_test_bow)

    return predictions

#takes in either a pre-processed text file for the training data (or an empty file if not pre-processed),
# and takes in the testing data to predict the labels for the test data
## INPUT::  1. Optional filename of preprocessed data 2. list of descriptions of properties
## OUTPUT::  Predicted labels for the properties
def predict_address(fileName, test_data):
    descriptions = []
    prices = []
    address = []
    processed_training_data = []
    labels = []

    if(fileName==''):
        print('No pre-processed file given.')
        with open('CleanedScraperOutput.csv','r') as csvfile:
            c = csv.reader(csvfile, delimiter = ',')
            next(csvfile)
            for row in c:
                processed_training_data.append(str(row[8]))
                prices.append(int(row[7]))
                address.append(str(row[2]))
        processed_training_data = text_process(processed_training_data)
        labels = address_labels(address)
        print('finished text processing and labelling')

        ## ensures there are at least 2 entries for each dublin area
        size_labels = [len([value for value in labels if value == area]) for area in range(-1,25)]
        adjusted_labels = labels
        for idx, num in enumerate(size_labels):
            if(num<2):
                for i, label in enumerate(labels):
                    if(label == idx - 1):
                        adjusted_labels.pop(i)
                        processed_training_data.pop(i)
        labels = adjusted_labels

        ## write this data to file for next time

        dict = {'Description': processed_training_data, 'AddressLabel': adjusted_labels}
        df = pd.DataFrame(dict)
        df.to_csv('ProcessedText.csv')

    else:
        print('Reading from pre-processed file: ', fileName)
        with open(fileName,'r') as csvfile:
            c = csv.reader(csvfile, delimiter = ',')
            next(csvfile)
            for row in c:
                processed_training_data.append(row[1])
                labels.append(int(row[2]))

    if(test_data!=[]):

        model_svm, tfidf = train_dataset(processed_training_data, labels)
        processed_test_data = text_process(test_data)
        predicted_labels = generate_svm_scores(processed_test_data, model_svm, tfidf)


        return predicted_labels



#################### EXAMPLE USAGE ###################


##### Option 1.
##if you have new data and just want to preprocess it for the future
predict_address('', [])


##### Option 2.
## I have data to get labels for in the file "testing.csv"
## (you just need the descriptions so you could get these any way you like
## this is just how i did it)
test_data = []
address = []
with open('testing.csv','r') as csvfile:
    c = csv.reader(csvfile, delimiter = ',')
    for row in c:
        test_data.append(str(row[8]))
        address.append(str(row[2]))   ## this line was just to check the predictions but isnt needed

## I then pass the descriptions as a list to the pipeline, along with the filename
## containing my already processed training data.
predictions = predict_address('ProcessedText.csv', test_data)
## predictions is now a list of the predicted addresses for the test data


##### Option 3.
## I have new training data and want to simultaneously predict some labels
new_predictions = predict_address('', test_data)
