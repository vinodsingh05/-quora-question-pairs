# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 15:30:11 2024

@author: vinodsingh
"""

import numpy as np
from fuzzywuzzy import fuzz
import pickle
import distance
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

import spacy




tfidf_scores = pickle.load(open('tfidf_score.pkl','rb'))
# tfidfw2vq2 = pickle.load(open('tfidf','rb'))



def preprocess(x): 
    x = str(x).lower().strip()
    # it replace string x step by step, first replace ',000,000' to 'm' after the result from first replaced string apply on the first replace string again ",000" to "k"
    x = x.replace(",000,000", "m").replace(",000", "k").replace("′", "'").replace("’", "'")\
                             .replace("won't", "will not").replace("cannot", "can not").replace("can't", "can not")\
                             .replace("n't", " not").replace("what's", "what is").replace("it's", "it is")\
                             .replace("'ve", " have").replace("i'm", "i am").replace("'re", " are")\
                             .replace("he's", "he is").replace("she's", "she is").replace("'s", " own")\
                             .replace("%", " percent ").replace("₹", " rupee ").replace("$", " dollar ")\
                             .replace("€", " euro ").replace("'ll", " will").replace("I've", "I have")
    
    # perform substitution or replacement of substrings within a string using regular expressions
    x = re.sub(r'([0-9]+)000000', r'\1m', x)
    x = re.sub(r'([0-9]+)000', r'\1k', x)
    
    # replacing the comma, dot, via space.
    x = re.sub('[^a-zA-Z]', ' ',x)
    
    # # Removing HTML tags
    # x = BeautifulSoup(x)
    # x = x.get_text()
    
    # Creating an object of PorterStemm
    ps = PorterStemmer()
    # ps.stem() this method will convert all verb into present tense
    x = ps.stem(x)
    
    
    return x
        

def test_common_words(q1,q2):
    w1 = set(map(lambda word: word.lower().strip(), q1.split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), q2.split(" ")))
    return len(w1 & w2)


def test_total_words(q1,q2):
    w1 = set(map(lambda word: word.lower().strip(), q1.split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), q2.split(" ")))
    return (len(w1) + len(w2))



def test_fetch_token_features(q1, q2):    
    
    STOP_WORDS = stopwords.words('english')
  
    # To get the results in 4 decemal points
    # we get division by zero that's why this
    SAFE_DIV = 0.0001
    
    # create a list which contains 8 elements, all initialized with 0.0
    token_features = [0.0]*8
  
  
    # converting sentence into tokens
    q1_tokens = q1.split()
    q2_tokens = q2.split()
    # print(q1_tokens, q2_tokens)
  
    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
          return token_features
  
    # Get the non-stopwords in Questions
    q1_words = set([word for word in q1_tokens if word not in STOP_WORDS])
    q2_words = set([word for word in q2_tokens if word not in STOP_WORDS])
  
    # Get the stopwords in Questions
    q1_stop_words = set([word for word in q1_tokens if word in STOP_WORDS])
    q2_stop_words = set([word for word in q2_tokens if word in STOP_WORDS])
  
    # Get the common non-stopwords from question pair
    common_word_count = len(q1_words.intersection(q2_words))
  
    # Get the common stopwords from question pair
    common_stopword_count = len(q1_stop_words.intersection(q2_stop_words))
  
    # Get the common token from question pari
    common_token_count = len(set(q1_tokens).intersection(set(q2_tokens)))
  
    # calculating CWC_min, CWC_max, CSC_min, CSC_max, CTC_min, CTC_max
    token_features[0] = common_word_count / (min(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[1] = common_word_count / (max(len(q1_words), len(q2_words))+ SAFE_DIV)
    token_features[2] = common_stopword_count / (min(len(q1_stop_words), len(q2_stop_words))+ SAFE_DIV)
    token_features[3] = common_stopword_count / (max(len(q1_stop_words), len(q2_stop_words))+ SAFE_DIV)
    token_features[4] = common_token_count / (min(len(q1_tokens), len(q2_tokens))+ SAFE_DIV)
    token_features[5] = common_token_count / (max(len(q1_tokens), len(q2_tokens))+ SAFE_DIV)
  
  
    # Last word of both question is same or not
    token_features[6] = int(q1_tokens[-1] == q2_tokens[-1])   # int because return True/False and int converts to 1/0
  
    # First word of both question is same or not
    token_features[7] = int(q1_tokens[0] == q2_tokens[0])
  
  
    return token_features



def test_fetch_length_features(q1, q2):
    length_features = [0.0] * 3

    # Converting the Sentence into Tokens:
    q1_tokens = q1.split()
    q2_tokens = q2.split()

    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return length_features

    # Absolute length features
    length_features[0] = abs(len(q1_tokens) - len(q2_tokens))

    # Average Token Length of both Questions
    length_features[1] = (len(q1_tokens) + len(q2_tokens)) / 2

    strs = [distance.lcsubstrings(q1, q2)]
    if len(strs) == 0:
        return 0
    else:
        length_features[2] = len(strs[0]) / (min(len(q1), len(q2)) + 1)

    return length_features



def test_fetch_fuzzy_features(q1, q2):
    fuzzy_features = [0.0] * 4

    # fuzz_ratio
    fuzzy_features[0] = fuzz.QRatio(q1, q2)

    # fuzz_partial_ratio
    fuzzy_features[1] = fuzz.partial_ratio(q1, q2)

    # token_sort_ratio
    fuzzy_features[2] = fuzz.token_sort_ratio(q1, q2)

    # token_set_ratio
    fuzzy_features[3] = fuzz.token_set_ratio(q1, q2)

    return fuzzy_features





def query_point_creator(q1, q2):
    input_query = []

    # preprocess
    q1 = preprocess(q1)
    q2 = preprocess(q2)

    # fetch basic features
    input_query.append(len(q1))
    input_query.append(len(q2))

    input_query.append(len(q1.split(" ")))
    input_query.append(len(q2.split(" ")))

    input_query.append(test_common_words(q1, q2))
    input_query.append(test_total_words(q1, q2))
    input_query.append(round(test_common_words(q1, q2) / test_total_words(q1, q2), 2))

    # fetch token features
    token_features = test_fetch_token_features(q1, q2)
    input_query.extend(token_features)

    # fetch length based features
    length_features = test_fetch_length_features(q1, q2)
    input_query.extend(length_features)

    # fetch fuzzy features
    fuzzy_features = test_fetch_fuzzy_features(q1, q2)
    input_query.extend(fuzzy_features)


    # merging both question to find tf-idf scores
    questions = [q1]+[q2]
    # finding tf-idf scores from tf-idfVetorizer
    tfidf_matrix = tfidf_scores.transform(questions)
    
    word2tfidf = dict(zip(tfidf_scores.get_feature_names_out(), tfidf_scores.idf_))
    # print(word2tfidf)
    
    nlp = spacy.load('en_core_web_lg')
    
    vecs1 = []
    doc1 = nlp(q1)
    mean_vec1 = np.zeros((len(doc1),len(doc1[0].vector)))
    for word1 in doc1:
        vec1 = word1.vector
        try:
            idf = word2tfidf[str(word1)]
        except:
            idf = 0
        mean_vec1 +=vec1*idf
    mean_vec1 = mean_vec1.mean(axis=0)
    vecs1.append(mean_vec1)
    
    vecs2 = []
    doc2 = nlp(q2)
    mean_vec2 = np.zeros((len(doc2),len(doc2[0].vector)))
    for word2 in doc2:
        vec2 = word2.vector
        try:
            idf = word2tfidf[str(word2)]
        except:
            idf = 0
        mean_vec2 +=vec2*idf
    mean_vec2 = mean_vec2.mean(axis=0)
    vecs2.append(mean_vec2)
    
    # print(len(input_query))
    return np.hstack((np.array(input_query).reshape(1,22), vecs1, vecs2))
    
    
    
    
# query_point_creator('hello vinod', 'suraj')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    