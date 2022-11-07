import pandas as pd
import csv
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import string
import re



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
    #print(vectoriser.get_feature_names_out())

    return vectoriser


def test_processing():
    descriptions = []
    processed_text = []
    with open('testing.csv','r') as csvfile:
        c = csv.reader(csvfile, delimiter = ',')

        next(csvfile)
        for row in c:
            descriptions.append(str(row[6]))

    for description in descriptions:
        if description!= '':
            processed_text.append(' '.join(text_process(description)))

    print('Number of non empty descriptions: ', len(processed_text))


    vectoriser = tf_idf(processed_text)


test_processing()
