# coding: utf-8

"""
CS579: Assignment 2

In this assignment, you will build a text classifier to determine whether a
movie review is expressing positive or negative sentiment. The data come from
the website IMDB.com.

You'll write code to preprocess the data in different ways (creating different
features), then compare the cross-validation accuracy of each approach. Then,
you'll compute accuracy on a test set and do some analysis of the errors.

The main method takes about 40 seconds for me to run on my laptop. Places to
check for inefficiency include the vectorize function and the
eval_all_combinations function.

Complete the 14 methods below, indicated by TODO.

As usual, completing one method at a time, and debugging with doctests, should
help.
"""

# No imports allowed besides these.
from collections import Counter, defaultdict
from itertools import chain, combinations
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from scipy.sparse import csr_matrix
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
import string
import tarfile
import urllib.request

#********* Extra Imports added for tuning.
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen
from sklearn.naive_bayes import MultinomialNB
import nltk
nltk.download("popular")


def download_data():
    """ Download and unzip data.
    DONE ALREADY.
    """
    url = 'https://www.dropbox.com/s/8oehplrobcgi9cq/imdb.tgz?dl=1'
    urllib.request.urlretrieve(url, 'imdb.tgz')
    tar = tarfile.open("imdb.tgz")
    tar.extractall()
    tar.close()


def read_data(path):
    """
    Walks all subdirectories of this path and reads all
    the text files and labels.
    DONE ALREADY.

    Params:
      path....path to files
    Returns:
      docs.....list of strings, one per document
      labels...list of ints, 1=positive, 0=negative label.
               Inferred from file path (i.e., if it contains
               'pos', it is 1, else 0)
    """
    fnames = sorted([f for f in glob.glob(os.path.join(path, 'pos', '*.txt'))])
    data = [(1, open(f).readlines()[0]) for f in sorted(fnames)]
    fnames = sorted([f for f in glob.glob(os.path.join(path, 'neg', '*.txt'))])
    data += [(0, open(f).readlines()[0]) for f in sorted(fnames)]
    data = sorted(data, key=lambda x: x[1])
    return np.array([d[1] for d in data]), np.array([d[0] for d in data])

tokenizer = ToktokTokenizer()
stopword_list = nltk.corpus.stopwords.words('english')
def remove_stopwords(text, is_lower_case=False):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text

def simple_stemmer(text):
    ps = nltk.porter.PorterStemmer()
    text = ' '.join([ps.stem(word) for word in text.split()])
    return text

def simple_lancaster(text):
    lancaster = LancasterStemmer()
    text = ' '.join([lancaster.stem(word) for word in text.split()])
    return text

def tokenize(doc, keep_internal_punct=False):
    """
    Tokenize a string.
    The string should be converted to lowercase.
    If keep_internal_punct is False, then return only the alphanumerics (letters, numbers and underscore).
    If keep_internal_punct is True, then also retain punctuation that
    is inside of a word. E.g., in the example below, the token "isn't"
    is maintained when keep_internal_punct=True; otherwise, it is
    split into "isn" and "t" tokens.

    Params:
      doc....a string.
      keep_internal_punct...see above
    Returns:
      a numpy array containing the resulting tokens.

    >>> tokenize(" Hi there! Isn't this fun?", keep_internal_punct=False)
    array(['hi', 'there', 'isn', 't', 'this', 'fun'], dtype='<U5')
    >>> tokenize("Hi there! Isn't this fun? ", keep_internal_punct=True)
    array(['hi', 'there', "isn't", 'this', 'fun'], dtype='<U5')
    """
    ###TODO
    punctuation = """!"#$%&'()*+,-./:;<=>?@[\]^`{|}~"""
    REPLACE_NO_SPACE = re.compile("[.;:!\'?,\"()\[\]]")
    REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")
    word_lst_no_punc = []
    doc = simple_stemmer(doc)
    doc = simple_lancaster(doc)
    if(keep_internal_punct):
        reviews = REPLACE_NO_SPACE.sub("", doc.lower())
        reviews = REPLACE_WITH_SPACE.sub(" ",reviews)
        word_lst = reviews.lower().split()
        words = ' '.join([word.strip(punctuation) for word in word_lst])
        words = re.sub(r'\s+'," ",words)
        return np.array(str.split(words))
    else:
        return np.array(re.sub("\W+",' ', doc.lower()).split())


def token_features(tokens, feats):
    """
    Add features for each token. The feature name
    is pre-pended with the string "token=".
    Note that the feats dict is modified in place,
    so there is no return value.

    Params:
      tokens...array of token strings from a document.
      feats....dict from feature name to frequency
    Returns:
      nothing; feats is modified in place.

    >>> feats = defaultdict(lambda: 0)
    >>> token_features(['hi', 'there', 'hi'], feats)
    >>> sorted(feats.items())
    [('token=hi', 2), ('token=there', 1)]
    """
    ###TODO
    for tok in tokens:
        if(feats["token="+tok] != None ):
            feats["token="+tok] = feats["token="+tok] + 1
        else:
            feats["token="+tok] = 1

def token_pair_features(tokens, feats, k=3):
    """
    Compute features indicating that two words occur near
    each other within a window of size k.

    For example [a, b, c, d] with k=3 will consider the
    windows: [a,b,c], [b,c,d]. In the first window,
    a_b, a_c, and b_c appear; in the second window,
    b_c, c_d, and b_d appear. This example is in the
    doctest below.
    Note that the order of the tokens in the feature name
    matches the order in which they appear in the document.
    (e.g., a__b, not b__a)

    Params:
      tokens....array of token strings from a document.
      feats.....a dict from feature to value
      k.........the window size (3 by default)
    Returns:
      nothing; feats is modified in place.

    >>> feats = defaultdict(lambda: 0)
    >>> token_pair_features(np.array(['a', 'b', 'c', 'd']), feats)
    >>> sorted(feats.items())
    [('token_pair=a__b', 1), ('token_pair=a__c', 1), ('token_pair=b__c', 2), ('token_pair=b__d', 1), ('token_pair=c__d', 1)]
    """
    ###TODO
    windows = []
    # Spliting into windows of size K and storing into windows array
    for i in range(0,len(tokens)):
        if(i+k <= len(tokens)):
            windows.append(tokens[i:i+k])
    
    # For each Window in windows , extracting pair and appending it to feats
    for window in windows:
        for i in range(0,len(window)):
            for j in range(i+1,len(window)):
                if(feats["token_pair="+window[i]+"__"+window[j]] != None ):
                    feats["token_pair="+window[i]+"__"+window[j]] += 1
                else:
                    feats["token_pair="+window[i]+"__"+window[j]] = 1

def getAFINNTerms():
    url = urlopen('http://www2.compute.dtu.dk/~faan/data/AFINN.zip') 
    zipfile = ZipFile(BytesIO(url.read()))
    afinn_file = zipfile.open('AFINN/AFINN-111.txt')

    afinn = dict()
    for line in afinn_file:
        parts = line.strip().split()
        #print(parts)
        if(len(parts) == 2):
            afinn[parts[0].decode("utf-8")] = int(parts[1])
    return afinn

#neg_words = set(['bad', 'hate', 'horrible', 'worst', 'boring'])
#pos_words = set(['awesome', 'amazing', 'best', 'good', 'great', 'love', 'wonderful'])
afinn_terms = getAFINNTerms()

def lexicon_features(tokens, feats):
    """
    Add features indicating how many time a token appears that matches either
    the neg_words or pos_words (defined above). The matching should ignore
    case.

    Params:
      tokens...array of token strings from a document.
      feats....dict from feature name to frequency
    Returns:
      nothing; feats is modified in place.

    In this example, 'LOVE' and 'great' match the pos_words,
    and 'boring' matches the neg_words list.
    >>> feats = defaultdict(lambda: 0)
    >>> lexicon_features(np.array(['i', 'LOVE', 'this', 'great', 'boring', 'movie']), feats)
    >>> sorted(feats.items())
    [('neg_words', 1), ('pos_words', 2)]
    """
    ###TODO
    feats["neg_words"] = 0
    positive_word_counter = 0
    feats["pos_words"] = 0
    negative_word_counter = 0
    for item in tokens:
        if(item.lower() in afinn_terms):
            if(afinn_terms[item.lower()] < 0):
                feats["neg_words"] += (-1 * afinn_terms[item.lower()])
                negative_word_counter += 1
        elif(item.lower() in afinn_terms):
            if(afinn_terms[item.lower()] > 0):
                feats["pos_words"] += (afinn_terms[item.lower()])
                positive_word_counter += 1
    if( negative_word_counter > 0 ):
        feats["neg_words"] = feats["neg_words"]/negative_word_counter
    if positive_word_counter > 0:
        feats["pos_words"] = feats["pos_words"]/positive_word_counter


def featurize(tokens, feature_fns):
    """
    Compute all features for a list of tokens from
    a single document.

    Params:
      tokens........array of token strings from a document.
      feature_fns...a list of functions, one per feature
    Returns:
      list of (feature, value) tuples, SORTED alphabetically
      by the feature name.

    >>> feats = featurize(np.array(['i', 'LOVE', 'this', 'great', 'movie']), [token_features, lexicon_features])
    >>> feats
    [('neg_words', 0), ('pos_words', 2), ('token=LOVE', 1), ('token=great', 1), ('token=i', 1), ('token=movie', 1), ('token=this', 1)]
    """
    ###TODO
    feats = defaultdict(lambda: 0)  
    for feature in feature_fns:
        feature(tokens, feats)
    return (sorted(feats.items()))


def vectorize(tokens_list, feature_fns, min_freq, vocab=None):
    """
    Given the tokens for a set of documents, create a sparse
    feature matrix, where each row represents a document, and
    each column represents a feature.

    Params:
      tokens_list...a list of lists; each sublist is an
                    array of token strings from a document.
      feature_fns...a list of functions, one per feature
      min_freq......Remove features that do not appear in
                    at least min_freq different documents.
    Returns:
      - a csr_matrix: See https://goo.gl/f5TiF1 for documentation.
      This is a sparse matrix (zero values are not stored).
      - vocab: a dict from feature name to column index. NOTE
      that the columns are sorted alphabetically (so, the feature
      "token=great" is column 0 and "token=horrible" is column 1
      because "great" < "horrible" alphabetically),

    When vocab is None, we build a new vocabulary from the given data.
    when vocab is not None, we do not build a new vocab, and we do not
    add any new terms to the vocabulary. This setting is to be used
    at test time.

    >>> docs = ["Isn't this movie great?", "Horrible, horrible movie"]
    >>> tokens_list = [tokenize(d) for d in docs]
    >>> feature_fns = [token_features]
    >>> X, vocab = vectorize(tokens_list, feature_fns, min_freq=1)
    >>> type(X)
    <class 'scipy.sparse.csr.csr_matrix'>
    >>> X.toarray()
    array([[1, 0, 1, 1, 1, 1],
           [0, 2, 0, 1, 0, 0]], dtype=int64)
    >>> sorted(vocab.items(), key=lambda x: x[1])
    [('token=great', 0), ('token=horrible', 1), ('token=isn', 2), ('token=movie', 3), ('token=t', 4), ('token=this', 5)]
    """
    ###TODO
    doc_feature_list = []
    feats = defaultdict(lambda: 0)
    rowIndices = []
    colIndices = []
    data = []
    
    #Looping over the list of tokens for each document
    for token in tokens_list:
        temp_feat = defaultdict()
        #Fetching the features for each document
        token_feature  = featurize(token, feature_fns)
        
        # Extracting the list of feature which are non zero
        for feature in token_feature:
            # Only Picking tokens whose occurance is one or more than once
            # Mainly to extract pos and neg values if no match found
            if(feature[1] >= 1):
                temp_feat[feature[0]] = feature[1]
                if feature[0] in feats:
                    feats[feature[0]] = feats[feature[0]] + 1
                else:
                    feats[feature[0]] = 1
        doc_feature_list.append(temp_feat)
    
    # Building The Vocabulary
    if vocab == None:
        vocab = {}
        for item in sorted(feats.items(), key=lambda x:x[0]):
            # Adding in the vocab list if the occurance of element is more than or equal to min frequency 
            if(item[1] >= min_freq):
                vocab[item[0]] = len(vocab)
    
    #Preparing the row, col and data from the extracted list of features 
    for index,feature in enumerate(doc_feature_list):
        for feat_item in feature:
            #Check If the col exists in vocab
            if feat_item in vocab:
                rowIndices.append(index)
                colIndices.append(vocab[feat_item])
                data.append(feature[feat_item])
    
    X = csr_matrix((np.array(data),(rowIndices, colIndices)))
    return X.astype(dtype=np.int64), vocab
    


def accuracy_score(truth, predicted):
    """ Compute accuracy of predictions.
    DONE ALREADY
    Params:
      truth.......array of true labels (0 or 1)
      predicted...array of predicted labels (0 or 1)
    """
    return len(np.where(truth==predicted)[0]) / len(truth)


def cross_validation_accuracy(clf, X, labels, k):
    """
    Compute the average testing accuracy over k folds of cross-validation. You
    can use sklearn's KFold class here (no random seed, and no shuffling
    needed).

    Params:
      clf......A LogisticRegression classifier.
      X........A csr_matrix of features.
      labels...The true labels for each instance in X
      k........The number of cross-validation folds.

    Returns:
      The average testing accuracy of the classifier
      over each fold of cross-validation.
    """
    ###TODO
    cv = KFold(n_splits=k)
    accuracies=[]
    for train_ind, test_ind in cv.split(X):
        clf.fit(X[train_ind], labels[train_ind])
        predictions = clf.predict(X[test_ind])
        accuracies.append(accuracy_score(labels[test_ind],predictions))
    return np.mean(accuracies)


def eval_all_combinations(docs, labels, punct_vals,
                          feature_fns, min_freqs):
    """
    Enumerate all possible classifier settings and compute the
    cross validation accuracy for each setting. We will use this
    to determine which setting has the best accuracy.

    For each setting, construct a LogisticRegression classifier
    and compute its cross-validation accuracy for that setting.

    In addition to looping over possible assignments to
    keep_internal_punct and min_freqs, we will enumerate all
    possible combinations of feature functions. So, if
    feature_fns = [token_features, token_pair_features, lexicon_features],
    then we will consider all 7 combinations of features (see Log.txt
    for more examples).

    Params:
      docs..........The list of original training documents.
      labels........The true labels for each training document (0 or 1)
      punct_vals....List of possible assignments to
                    keep_internal_punct (e.g., [True, False])
      feature_fns...List of possible feature functions to use
      min_freqs.....List of possible min_freq values to use
                    (e.g., [2,5,10])

    Returns:
      A list of dicts, one per combination. Each dict has
      four keys:
      'punct': True or False, the setting of keep_internal_punct
      'features': The list of functions used to compute features.
      'min_freq': The setting of the min_freq parameter.
      'accuracy': The average cross_validation accuracy for this setting, using 5 folds.

      This list should be SORTED in descending order of accuracy.

      This function will take a bit longer to run (~20s for me).
    """
    ###TODO
    result = []
    features_fns_list = []
    for pos in range(0,len(feature_fns)):
        features_fns_list.append([feature_fns[pos]])
        for j in range(pos,len(feature_fns)):
            if(pos != j):
                features_fns_list.append([feature_fns[pos],feature_fns[j]])
    features_fns_list.append(feature_fns)
    
    for punct_val in punct_vals:
        tokens_list = []
        for doc in docs:
            tokens_list.append(tokenize(str(doc).lower(), punct_val))
        for min_fq in min_freqs:
            for feature in features_fns_list:
                X, t_vocab = vectorize(tokens_list, feature, min_freq=min_fq)
                #model = LogisticRegression()
                model = MultinomialNB()
                accuracy = cross_validation_accuracy(model, X, labels, 5)
                res_dict = {}
                res_dict["punct"] = punct_val
                res_dict["features"] = feature
                res_dict["min_freq"] = min_fq
                res_dict["accuracy"] = accuracy
                result.append(res_dict)
    
    return sorted(result, key=lambda x:(-x['accuracy'],-x['min_freq']))


def plot_sorted_accuracies(results):
    """
    Plot all accuracies from the result of eval_all_combinations
    in ascending order of accuracy.
    Save to "accuracies.png".
    """
    ###TODO
    x_axis = []
    y_axis = []
    for index,res in enumerate(results):
        x_axis.append(res["accuracy"])
        y_axis.append(len(results)-index)
    
    plt.ylabel("accuracy")
    plt.xlabel("setting")
    plt.plot(y_axis,x_axis)
    plt.savefig("accuracies.png")



def mean_accuracy_per_setting(results):
    """
    To determine how important each model setting is to overall accuracy,
    we'll compute the mean accuracy of all combinations with a particular
    setting. For example, compute the mean accuracy of all runs with
    min_freq=2.

    Params:
      results...The output of eval_all_combinations
    Returns:
      A list of (accuracy, setting) tuples, SORTED in
      descending order of accuracy.
    """
    ###TODO
    mean_accuracy = defaultdict() # total Accuracy
    no_of_terms = defaultdict() # Count for No of Terms Added to the dict key
    mean_acc_per_setting = []
    for result in results:
        key_min_freq = "min_freq="+str(result["min_freq"])
        fun_name = ""
        for name in [res.__name__ for res in result["features"]]:
            fun_name = fun_name + " " + name
        key_features = "features="+fun_name.strip()
        key_punct = "punct="+str(result["punct"])
        if(key_min_freq in mean_accuracy):
            mean_accuracy[key_min_freq] += result["accuracy"]
            no_of_terms[key_min_freq] += 1
        else:
            mean_accuracy[key_min_freq] = result["accuracy"]
            no_of_terms[key_min_freq] = 1
            
        if(key_features in mean_accuracy):
            mean_accuracy[key_features] += result["accuracy"]
            no_of_terms[key_features] += 1
        else:
            mean_accuracy[key_features] = result["accuracy"]
            no_of_terms[key_features] = 1
            
        if(key_punct in mean_accuracy):
            mean_accuracy[key_punct] += result["accuracy"]
            no_of_terms[key_punct] += 1
        else:
            mean_accuracy[key_punct] = result["accuracy"]
            no_of_terms[key_punct] = 1

    for mean_acc_combination in mean_accuracy:
        mean_acc_per_setting.append(((mean_accuracy[mean_acc_combination]/no_of_terms[mean_acc_combination]),mean_acc_combination))
    return sorted(mean_acc_per_setting, key=lambda x:-x[0])


def fit_best_classifier(docs, labels, best_result):
    """
    Using the best setting from eval_all_combinations,
    re-vectorize all the training data and fit a
    LogisticRegression classifier to all training data.
    (i.e., no cross-validation done here)

    Params:
      docs..........List of training document strings.
      labels........The true labels for each training document (0 or 1)
      best_result...Element of eval_all_combinations
                    with highest accuracy
    Returns:
      clf.....A LogisticRegression classifier fit to all
            training data.
      vocab...The dict from feature name to column index.
    """
    ###TODO
    tokens_list = []
    for doc in docs:
        tokens_list.append(tokenize(str(doc).lower(), best_result["punct"]))
    X, t_vocab = vectorize(tokens_list, best_result["features"], best_result["min_freq"])
    model = LogisticRegression()
    model.fit(X,labels)
    return model,t_vocab


def top_coefs(clf, label, n, vocab):
    """
    Find the n features with the highest coefficients in
    this classifier for this label.
    See the .coef_ attribute of LogisticRegression.

    Params:
      clf.....LogisticRegression classifier
      label...1 or 0; if 1, return the top coefficients
              for the positive class; else for negative.
      n.......The number of coefficients to return.
      vocab...Dict from feature name to column index.
    Returns:
      List of (feature_name, coefficient) tuples, SORTED
      in descending order of the coefficient for the
      given class label.
    """
    ###TODO
    top_results = [] # For capturing the feature name and coefficient
    coef = clf.coef_[0]
    top_neg_coef = []
    top_pos_coef = []
    if(label == 1):
        #*** TOP POSITIVE COEFFICIENT 
        #Sorted Vocab to match the position with the index
        for index,feature in enumerate(sorted(vocab)):
            value = coef[index]
            top_results.append((feature, value))
        top_pos_coef = sorted(top_results, key=lambda x:-x[1])[:n]
    elif(label == 0):
        #*** TOP Negative COEFFICIENT 
        for index,feature in enumerate(sorted(vocab)):
            value = coef[index]
            top_results.append((feature, value))
        for neg_coef in sorted(top_results, key=lambda x:x[1])[:n]:
            top_neg_coef.append((neg_coef[0],-1* neg_coef[1]))
    if(label == 0):
      return top_neg_coef
    else:
      if(label == 1):
        return top_pos_coef


def parse_test_data(best_result, vocab):
    """
    Using the vocabulary fit to the training data, read
    and vectorize the testing data. Note that vocab should
    be passed to the vectorize function to ensure the feature
    mapping is consistent from training to testing.

    Note: use read_data function defined above to read the
    test data.

    Params:
      best_result...Element of eval_all_combinations
                    with highest accuracy
      vocab.........dict from feature name to column index,
                    built from the training data.
    Returns:
      test_docs.....List of strings, one per testing document,
                    containing the raw.
      test_labels...List of ints, one per testing document,
                    1 for positive, 0 for negative.
      X_test........A csr_matrix representing the features
                    in the test data. Each row is a document,
                    each column is a feature.
    """
    ###TODO
    tokens_list = []
    test_docs, test_labels = read_data(os.path.join('data', 'test'))
    for doc in test_docs:
        tokens_list.append(tokenize(str(doc).lower(), best_result["punct"]))
    X_test, t_vocab = vectorize(tokens_list, best_result["features"], best_result["min_freq"], vocab)
    
    return test_docs,test_labels,X_test


def print_top_misclassified(test_docs, test_labels, X_test, clf, n):
    """
    Print the n testing documents that are misclassified by the
    largest margin. By using the .predict_proba function of
    LogisticRegression <https://goo.gl/4WXbYA>, we can get the
    predicted probabilities of each class for each instance.
    We will first identify all incorrectly classified documents,
    then sort them in descending order of the predicted probability
    for the incorrect class.
    E.g., if document i is misclassified as positive, we will
    consider the probability of the positive class when sorting.

    Params:
      test_docs.....List of strings, one per test document
      test_labels...Array of true testing labels
      X_test........csr_matrix for test data
      clf...........LogisticRegression classifier fit on all training
                    data.
      n.............The number of documents to print.

    Returns:
      Nothing; see Log.txt for example printed output.
    """
    ###TODO
    temp = []
    top_misclassified_list = {}
    predicted_probability = clf.predict_proba(X_test)
    predicted_values = clf.predict(X_test)
    for index,value in enumerate(clf.predict(X_test)):
        if(predicted_values[index] != test_labels[index]):
            misclassified_element = {}
            misclassified_element["truth"] = test_labels[index]
            misclassified_element["predicted"] = predicted_values[index]
            misclassified_element["proba"] = predicted_probability[index][value]
            misclassified_element["doc"] = test_docs[index]
            temp.append(predicted_probability[index][value])
            top_misclassified_list[index]=(misclassified_element)
    for document in sorted(list(top_misclassified_list.items()), key=lambda x:-x[1]['proba'])[:5]:
        print('truth=%s predicted=%s proba=%.6f' %  (str(document[1]["truth"]),str(document[1]["predicted"]),document[1]["proba"]))
        print(str(document[1]["doc"]) + "\n")


def main():
    """
    Put it all together.
    ALREADY DONE.
    """
    feature_fns = [token_features, token_pair_features, lexicon_features]
    # Download and read data.
    download_data()
    docs, labels = read_data(os.path.join('data', 'train'))
    # Evaluate accuracy of many combinations
    # of tokenization/featurization.
    results = eval_all_combinations(docs, labels,
                                    [True, False],
                                    feature_fns,
                                    [2,5,10])
    # Print information about these results.
    best_result = results[0]
    worst_result = results[-1]
    print('best cross-validation result:\n%s' % str(best_result))
    print('worst cross-validation result:\n%s' % str(worst_result))
    plot_sorted_accuracies(results)

    print('\nMean Accuracies per Setting:')
    print('\n'.join(['%s: %.5f' % (s,v) for v,s in mean_accuracy_per_setting(results)]))

    # Fit best classifier.
    clf, vocab = fit_best_classifier(docs, labels, results[0])

    # Print top coefficients per class.
    print('\nTOP COEFFICIENTS PER CLASS:')
    print('negative words:')
    print('\n'.join(['%s: %.5f' % (t,v) for t,v in top_coefs(clf, 0, 5, vocab)]))
    print('\npositive words:')
    print('\n'.join(['%s: %.5f' % (t,v) for t,v in top_coefs(clf, 1, 5, vocab)]))

    # Parse test data
    test_docs, test_labels, X_test = parse_test_data(best_result, vocab)

    # Evaluate on test set.
    predictions = clf.predict(X_test)
    print('testing accuracy=%f' %
          accuracy_score(test_labels, predictions))

    print('\nTOP MISCLASSIFIED TEST DOCUMENTS:')
    print_top_misclassified(test_docs, test_labels, X_test, clf, 5)


if __name__ == '__main__':
    main()
