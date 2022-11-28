import pandas as pd
import csv
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import string
import re
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn import svm
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score, f1_score
poppedindexes = []
test_data = []

SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16


plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


# takes in the Dublin postcode and returns the score
def address_labels(address):

    labels = []

    for label in address:
        if(label == 'Other' or label == '' or label == '6' or label == '14' or label == '16' or label == '18'):
            labels.append(5)
        elif(label =='6w' or label == '6W' or label == '12' or label == '10' or label == '20' or label == '22'):
            labels.append(4)
        elif(label == '1' or label == '2' or label == '4' or label == '7' or label == '8'):
            labels.append(1)
        elif(label == '11' or label == '15' or label == '9' or label == '17'):
            labels.append(2)
        elif(label == '3' or label == '5' or label == '13'):
            labels.append(3)
        elif(label == '11' or label == '15' or label == '22' or label == '24'):
            labels.append(6)


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


#this function was used to perform cross validation on the sentiment analysis model
def cross_val():
    processed_training_data = []
    labels = []
    #get data
    print('Reading from pre-processed file: ', 'ProcessedText.csv')
    with open('ProcessedText.csv','r') as csvfile:
        c = csv.reader(csvfile, delimiter = ',')
        next(csvfile)
        for row in c:
            processed_training_data.append(row[1])
            labels.append(int(row[2]))

    df_values = [1,2,3,4,5]
    c = 8
    predictions = []
    mean_f1=[]
    std_acc = []

    #first test the min_df
    for df in df_values:
        model_svm, tfidf_vec, X_train_bow = train_dataset(processed_training_data, labels, df, c)
        temp = []
        kf = KFold(n_splits=10)

        processed_training_data = np.array(processed_training_data)
        labels = np.array(labels)
        for train, test in kf.split(processed_training_data):
            X_train_bow = tfidf_vec.transform(processed_training_data[train]) # fit train
            model_svm.fit(X_train_bow, labels[train])
            X_test_bow = tfidf_vec.transform(processed_training_data[test]) # fit train
            ypred = model_svm.predict(X_test_bow)
            temp.append(f1_score(labels[test], ypred, average='weighted'))
        mean_f1.append(np.array(temp).mean())
        std_acc.append(np.array(temp).std())


    plt.errorbar(df_values, mean_f1, yerr=std_acc, marker = 's', ms = 5, ecolor='r', color = 'k')
    plt.legend()
    plt.xlabel(r'Cut off ($min\_df$)')
    plt.xticks(df_values)
    plt.ylabel('F1 Score')
    plt.title('Cross-validation for the TfIDF vectorizer')
    plt.show()


    #then setting df to be ==2 based on the cross validation
    c_values = [0.5, 2, 4, 6, 8, 10]
    predictions = []
    mean_f1=[]
    std_acc = []

    df = 2


    #first test the min_df
    for c in c_values:
        model_svm, tfidf_vec, X_train_bow = train_dataset(processed_training_data, labels, df, c)
        temp = []
        kf = KFold(n_splits=10)

        processed_training_data = np.array(processed_training_data)
        labels = np.array(labels)
        for train, test in kf.split(processed_training_data):
            X_train_bow = tfidf_vec.transform(processed_training_data[train]) # fit train
            model_svm.fit(X_train_bow, labels[train])
            X_test_bow = tfidf_vec.transform(processed_training_data[test]) # fit train
            ypred = model_svm.predict(X_test_bow)
            temp.append(f1_score(labels[test], ypred, average='weighted'))
        mean_f1.append(np.array(temp).mean())
        std_acc.append(np.array(temp).std())

            #    model_svm.fit(X_train_bow, y_train)

        #we will look at the overall accuracy
    plt.errorbar(c_values, mean_f1, yerr=std_acc, marker = 's', ms = 5, ecolor='r', color = 'k')
    plt.legend()
    plt.xlabel(r'Penalty Weighting, $C$')
    plt.xticks(c_values)
    plt.ylabel('F1 Score')
    plt.title('Cross-validation for the SVM')
    plt.show()
    return 0

## takes in the already processed training data and trains the svm. Returns the trained svm and tfidf.
def train_dataset(processed_training_data, labels, df, c):

    X_train = processed_training_data
    y_train = labels

    tfidf_vec = TfidfVectorizer(min_df = df, token_pattern = r'[a-zA-Z]+')  #1
    X_train_bow = tfidf_vec.fit_transform(X_train) # fit train
    model_svm = svm.SVC(C=c, kernel='linear')  #8
    model_svm.fit(X_train_bow, y_train)

    return model_svm, tfidf_vec, X_train_bow


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
    addressFull = []
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
                addressFull.append(str(row[1]))
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
                        addressFull.pop(i)
                        poppedindexes.append(i)
                        #print('i:', i)
        labels = adjusted_labels

        ## write this data to file for next time
        dict = {'Description': processed_training_data, 'AddressLabel': adjusted_labels, 'Address': addressFull}
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
        # pop pop
        for index in poppedindexes:
            test_data.remove(test_data[index])

        #training
        model_svm, tfidf, _ = train_dataset(processed_training_data, labels, 2, 8)
        processed_test_data = text_process(test_data)
        print('length of processed_test_data: ', len(processed_test_data))
        predicted_labels = generate_svm_scores(processed_test_data, model_svm, tfidf)


        return predicted_labels


def neuralData():
    #Read data from original scraper, add it to new csv file
    print('Reading from scraped data: CleanedScraperOutput.csv')
    results = pd.read_csv('CleanedScraperOutput.csv')
    print('Number of lines in CleanedScraperOutput CSV: ', len(results))

    print('Converting CSV columns into lists:')
    address = results['Address'].tolist()
    location = results['dublinRegionString'].tolist()
    location = [str(s).replace('Other', '0') for s in location]
    location = [str(s).replace('6W', '-1') for s in location]
    location = [str(s).replace('6w', '1') for s in location]
    location = [str(s).replace('nan', '0') for s in location] ###<===============why not cuaght before !?;""
    location = [int(t) for t in location]

    bedroom = results['Bedroom'].tolist()
    bathroom = results['Bathroom'].tolist()
    propertyType = results['PropertyType'].tolist() #Need to convert the three different types into numbers
    propertyType = [s.replace('House', '2') for s in propertyType]
    propertyType = [s.replace('Apartment', '1') for s in propertyType]
    propertyType = [s.replace('Studio', '0') for s in propertyType]
    propertyType = [int(t) for t in propertyType]
    # for i in range(len(propertyType)):
    #     if (propertyType[i] == 'House'):
    #         propertyType2.append(2)
    #     elif (propertyType[i] == 'Apartment'):
    #         propertyType2.append(1)
    #     else:
    #         propertyType2.append(0)
    #print(propertyType2)

    price = results['Price'].tolist()
    print('...Done.')

    #Read and/or get sentiment analysis class, add it in
    print('Receiving predictions...')
    predictions = predict_address('', test_data)
    #predictions = predict_address('ProcessedText.csv', test_data)
    print(predictions)
    print('Reading from processed text and utilising SVM model for classification: CleanedScraperOutput.csv')
    results = pd.read_csv('ProcessedText.csv')
    print('Number of lines in ProcessedText CSV: ', len(results))
    processedTextAddress = results['Address'].tolist()
    processedTextAddressLabel = results['AddressLabel'].tolist()
    processedText = results['Description'].tolist()
    print('...Done.')


    print('\nTraining SVM')
    print('Determine if entries are equivalent')
    print('Size of address list:', len(address))
    print('Size of processed address list:', len(processedTextAddress))

    if len(address) != len(processedTextAddress):
        print('\nfiltering out miscellaneous addresses..')
        print(poppedindexes)
        poppedindexes.sort()
        print(type(address))
        for index in poppedindexes:
            print(index)
            print(address[index])
            address.remove(address[index])
            location.remove(location[index])
            bedroom.remove(bedroom[index])
            bathroom.remove(bathroom[index])
            propertyType.remove(propertyType[index])
            price.remove(price[index])
        print('Size of address list:', len(address))
        print('Size of processed address list:', len(processedTextAddress))
        print('Re-determine if entries are equivalent')
        print('Size of address list:', len(address))
        print('Size of processed address list:', len(processedTextAddress))


    print('Length of predictions: ', len(predictions))
    print('...Done.')


    ## write this data to file for next time
    print('Placing data into new CSV file')
    dict = {'Address': address, 'Location': location, 'Bedroom': bedroom, 'Bathroom': bathroom, 'PropertyType': propertyType, 'Price': price, 'Predictions': predictions}
    df = pd.DataFrame(dict)
    df.to_csv('neuralData.csv')
    print('...Done.')

with open('CleanedScraperOutput.csv','r') as csvfile:
    c = csv.reader(csvfile, delimiter = ',')
    for row in c:

        if row[8] != 'Description':
            test_data.append(str(row[8]))

#fig, axs = plt.subplots(1,2, sharey=True)

#Create new CSV file for neural data
neuralData()
#cross_val()
