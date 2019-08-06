# coding: utf-8

# # Assignment 3:  Recommendation systems
#
# Here we'll implement a content-based recommendation algorithm.
# It will use the list of genres for a movie as the content.
# The data come from the MovieLens project: http://grouplens.org/datasets/movielens/
# Note that I have not provided many doctests for this one. I strongly
# recommend that you write your own for each function to ensure your
# implementation is correct.

# Please only use these imports.
from collections import Counter, defaultdict
import math
import numpy as np
import os
import pandas as pd
import re
from scipy.sparse import csr_matrix
import urllib.request
import zipfile

def download_data():
    """ DONE. Download and unzip data.
    """
    url = 'https://www.dropbox.com/s/p9wmkvbqt1xr6lc/ml-latest-small.zip?dl=1'
    urllib.request.urlretrieve(url, 'ml-latest-small.zip')
    zfile = zipfile.ZipFile('ml-latest-small.zip')
    zfile.extractall()
    zfile.close()


def tokenize_string(my_string):
    """ DONE. You should use this in your tokenize function.
    """
    return re.findall('[\w\-]+', my_string.lower())


def tokenize(movies):
    """
    Append a new column to the movies DataFrame with header 'tokens'.
    This will contain a list of strings, one per token, extracted
    from the 'genre' field of each movie. Use the tokenize_string method above.

    Note: you may modify the movies parameter directly; no need to make
    a new copy.
    Params:
      movies...The movies DataFrame
    Returns:
      The movies DataFrame, augmented to include a new column called 'tokens'.

    >>> movies = pd.DataFrame([[123, 'Horror|Romance'], [456, 'Sci-Fi']], columns=['movieId', 'genres'])
    >>> movies = tokenize(movies)
    >>> movies['tokens'].tolist()
    [['horror', 'romance'], ['sci-fi']]
    """
    ###TODO
    movies_genres = []
    for movie_genres in movies.genres:
      movies_genres.append(tokenize_string(movie_genres))
    # Assigning the values to the token field and ensuring that the values are placed in the appropriate field.
    movies['tokens'] = pd.Series(movies_genres, index = movies.index)
    return movies



def featurize(movies):
    """
    Append a new column to the movies DataFrame with header 'features'.
    Each row will contain a csr_matrix of shape (1, num_features). Each
    entry in this matrix will contain the tf-idf value of the term, as
    defined in class:
    tfidf(i, d) := tf(i, d) / max_k tf(k, d) * log10(N/df(i))
    where:
    i is a term
    d is a document (movie)
    tf(i, d) is the frequency of term i in document d
    max_k tf(k, d) is the maximum frequency of any term in document d
    N is the number of documents (movies)
    df(i) is the number of unique documents containing term i

    Params:
      movies...The movies DataFrame
    Returns:
      A tuple containing:
      - The movies DataFrame, which has been modified to include a column named 'features'.
      - The vocab, a dict from term to int. Make sure the vocab is sorted alphabetically as in a2 (e.g., {'aardvark': 0, 'boy': 1, ...})
    
    >>> movies = pd.DataFrame([[111, 'comedy|comedy|thriller|romance|horror',['comedy', 'comedy', 'thriller', 'romance', 'horror']],[222, 'romance', ['romance']]], columns=['movieId', 'genres', 'tokens'])
    >>> movies, vocab = featurize(movies)
    >>> sorted(vocab.items())
    [('comedy', 0), ('horror', 1), ('romance', 2), ('thriller', 3)]

    >>> movies = pd.DataFrame([[111, 'comedy|comedy|thriller|romance|horror',['comedy', 'comedy', 'thriller', 'romance', 'horror']],[222, 'romance', ['romance']]], columns=['movieId', 'genres', 'tokens'])
    >>> movies, vocab = featurize(movies)
    >>> row0 = movies['features'].tolist()[0]
    >>> '{:.2f}'.format(round(max(list(row0.data)), 2))
    '0.30'
    >>> '{:.1f}'.format(round(min(row0.toarray()[0]), 1))
    '0.0'
    """
    ###TODO
    tf = dict()
    # Dictionary for capturing the unique word 
    df = dict()
    temp_vocab = []
    vocab = dict()
    for tokens in movies.tokens:
        # Created token set for holding unique terms for a document and show not contain repeated terms
        token_set = set()
        for token in tokens:
            token_set.add(token)
            if token not in temp_vocab:
                temp_vocab.append(token)
        for token in token_set:
            if( token not in df):
                df[token] = 1
            else:
                df[token] += 1 
    #Capturing the sorted vocabulary
    for v in sorted(temp_vocab):
        pos = len(vocab)
        vocab[v] = pos
    # Array for capturing all the csr_matrix for all features
    movies_features = []
    for tokens in movies.tokens:
        col = []
        row = []
        featureVectData = []
        tf.clear()
        for token in tokens:
            if( token not in tf):
                tf[token] = 1
            else:
                tf[token] += 1
        for token in set(tokens):
            if token in vocab:
                col.append(vocab[token])
                row.append(0)
                #tf(i, d) / max_k tf(k, d) * log10(N/df(i))
                tfidf = tf[token] / max(tf.values()) * math.log10(len(movies) / df[token])
                #print(tfidf)
                featureVectData.append(tfidf)
        #print("------------")
        matrix = csr_matrix((featureVectData,(row,col)),shape=(1,len(vocab)))
        movies_features.append(matrix)
    movies['features'] = pd.Series(movies_features, index = movies.index)
    return movies,vocab


def train_test_split(ratings):
    """DONE.
    Returns a random split of the ratings matrix into a training and testing set.
    """
    test = set(range(len(ratings))[::1000])
    train = sorted(set(range(len(ratings))) - test)
    test = sorted(test)
    return ratings.iloc[train], ratings.iloc[test]


def cosine_sim(a, b):
    """
    Compute the cosine similarity between two 1-d csr_matrices.
    Each matrix represents the tf-idf feature vector of a movie.
    Params:
      a...A csr_matrix with shape (1, number_features)
      b...A csr_matrix with shape (1, number_features)
    Returns:
      A float. The cosine similarity, defined as: dot(a, b) / ||a|| * ||b||
      where ||a|| indicates the Euclidean norm (aka L2 norm) of vector a.
    """
    ###TODO
    dot_a_b = np.dot(a.toarray(), np.transpose(b.toarray()))
    norm_a = np.linalg.norm(a.toarray())
    norm_b = np.linalg.norm(b.toarray())
    sim = dot_a_b / (norm_a * norm_b)
    return sim[0][0]


def make_predictions(movies, ratings_train, ratings_test):
    """
    Using the ratings in ratings_train, predict the ratings for each
    row in ratings_test.

    To predict the rating of user u for movie i: Compute the weighted average
    rating for every other movie that u has rated.  Restrict this weighted
    average to movies that have a positive cosine similarity with movie
    i. The weight for movie m corresponds to the cosine similarity between m
    and i.

    If there are no other movies with positive cosine similarity to use in the
    prediction, use the mean rating of the target user in ratings_train as the
    prediction.

    Params:
      movies..........The movies DataFrame.
      ratings_train...The subset of ratings used for making predictions. These are the "historical" data.
      ratings_test....The subset of ratings that need to predicted. These are the "future" data.
    Returns:
      A numpy array containing one predicted rating for each element of ratings_test.
    
    >>> movies = pd.DataFrame([[111, 'adventure|romance', ['adventure', 'romance']],[222, 'comedy|crime', ['comedy', 'crime']],[333, 'drama', ['drama']],[000, 'fantasy', ['fantasy']]], columns=['movieId', 'genres', 'tokens'])
    >>> movies, vocab = featurize(movies)
    >>> ratings_train = pd.DataFrame([[11, 111, 2, 1554321200],[11, 222, 3.5, 1554321201],[11, 333, 3, 1554321202],[12, 111, 4.5, 1554321203],[12, 222, 3.5, 1554321204],[12, 333, 2, 1554321205],[13, 111, 5, 1554321206],[13, 222, 1.5, 1554321207]],columns=['userId', 'movieId', 'rating', 'timestamp'])
    >>> ratings_test = pd.DataFrame([[13, 333, 4, 1260759152]],columns=['userId', 'movieId', 'rating', 'timestamp'])
    >>> round(make_predictions(movies, ratings_train, ratings_test)[0],1)
    3.2
    
    """
    ###TODO
    rating_list = []
    user_ids = sorted(set(ratings_train.userId))
    cosine_sim_all = []
    #print(len(ratings_train),len(ratings_test))
    for item in zip(ratings_test.userId,ratings_test.movieId) :
        numerator = 0
        denominator = 0
        user_sim = []
        result = 0
        # Flag for checking sim positive value 0 for negative value and 1 for positive value
        sim_pos_val = 0
        movie_1 = movies[movies.movieId == item[1]]['features'].iloc[0]
        for index, row in ratings_train[ratings_train.userId == item[0]].iterrows():
            if row.movieId != item[1]:
                movie_2 = movies[movies.movieId == row.movieId]['features'].iloc[0]
                sim = cosine_sim(movie_1, movie_2)
                user_sim.append(row.rating)
                if sim > 0 :
                    sim_pos_val = 1
                    numerator += sim * row.rating
                    denominator += sim
        
        if(sim_pos_val == 1):
            result = numerator/denominator
        else:
            result = np.mean(user_sim)
        rating_list.append(result)
    return np.asarray(rating_list)


def mean_absolute_error(predictions, ratings_test):
    """DONE.
    Return the mean absolute error of the predictions.
    """
    return np.abs(predictions - np.array(ratings_test.rating)).mean()


def main():
    download_data()
    path = 'ml-latest-small'
    ratings = pd.read_csv(path + os.path.sep + 'ratings.csv')
    movies = pd.read_csv(path + os.path.sep + 'movies.csv')
    movies = tokenize(movies)
    movies, vocab = featurize(movies)
    print('vocab:')
    print(sorted(vocab.items())[:10])
    ratings_train, ratings_test = train_test_split(ratings)
    print('%d training ratings; %d testing ratings' % (len(ratings_train), len(ratings_test)))
    predictions = make_predictions(movies, ratings_train, ratings_test)
    print('error=%f' % mean_absolute_error(predictions, ratings_test))
    print(predictions[:10])


if __name__ == '__main__':
    main()
