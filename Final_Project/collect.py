"""
Collect data.
"""
from collections import Counter
import matplotlib.pyplot as plt
import networkx as nx
import sys
import time
import pickle
import os
import json
from TwitterAPI import TwitterAPI
import xlsxwriter


#First Acc
consumer_key = '7DXeFhtbVZD03as0CcDV21wlO'
consumer_secret = 'Q2P6veXKWQsp8ZycvnLCqrcx2rPUOgQ0BbSkvkeyExeqsrHwTG'
access_token = '724196648331530241-jiTT8Skq9GHlwKSPWpQGuNBo77vt89c'
access_token_secret = '9wQrdJ5IliMUzdLQer4wEmiTMCUFTHdMwyIwFVb40s0G4'
'''

#2nd Acc
consumer_key = 'x9hodJQTQErWnOnKv66PFTqKv'
consumer_secret = 'lkvdHIaUnvYr90gySUJZOj4NNhiHNmCCyWQ1xpE061USJvUknE'
access_token = '155951719-zfuC0is9UyVUjZUeBU8ECvfYGdiR93b6F8V9YXh9'
access_token_secret = 'IJoqHINaCt5HtsDf4QU3bKuhDxzVUwfAKzVecEDGAqIno'
'''

def get_twitter():
    """ Construct an instance of TwitterAPI using the tokens you entered above.
    Returns:
      An instance of TwitterAPI.
    """
    return TwitterAPI(consumer_key, consumer_secret, access_token, access_token_secret)


def read_screen_names(filename):
    """
    Read a text file containing Twitter screen_names, one per line.

    Params:
        filename....Name of the file to read.
    Returns:
        A list of strings, one per screen_name, in the order they are listed
        in the file.

    Here's a doctest to confirm your implementation is correct.
    >>> read_screen_names('candidates.txt')
    ['DrJillStein', 'GovGaryJohnson', 'HillaryClinton', 'realDonaldTrump']
    """
    ###TODO
    text_file = open("users.txt", "r")
    candidates = text_file.readlines()
    candidates = [item.replace("\n", "") for item in candidates]
    return candidates


# I've provided the method below to handle Twitter's rate limiting.
# You should call this method whenever you need to access the Twitter API.
def robust_request(twitter, resource, params, max_tries=5):
    """ If a Twitter request fails, sleep for 15 minutes.
    Do this at most max_tries times before quitting.
    Args:
      twitter .... A TwitterAPI object.
      resource ... A resource string to request; e.g., "friends/ids"
      params ..... A parameter dict for the request, e.g., to specify
                   parameters like screen_name or count.
      max_tries .. The maximum number of tries to attempt.
    Returns:
      A TwitterResponse object, or None if failed.
    """
    for i in range(max_tries):
        request = twitter.request(resource, params)
        if request.status_code == 200:
            return request
        else:
            print('Got error %s \nsleeping for 15 minutes.' % request.text)
            sys.stderr.flush()
            time.sleep(61 * 15)

def get_users(twitter, screen_names):
    """Retrieve the Twitter user objects for each screen_name.
    Params:
        twitter........The TwitterAPI object.
        screen_names...A list of strings, one per screen_name
    Returns:
        A list of dicts, one per user, containing all the user information
        (e.g., screen_name, id, location, etc)

    See the API documentation here: https://dev.twitter.com/rest/reference/get/users/lookup

    In this example, I test retrieving two users: twitterapi and twitter.

    >>> twitter = get_twitter()
    >>> users = get_users(twitter, ['twitterapi', 'twitter'])
    >>> [u['id'] for u in users]
    [6253282, 783214]
    """
    ###TODO
    request = robust_request(twitter,'users/lookup', {'screen_name': screen_names })
    users = [r for r in request]
    return users


def get_friends(twitter, screen_name):
    """ Return a list of Twitter IDs for users that this person follows, up to 5000.
    See https://dev.twitter.com/rest/reference/get/friends/ids

    Note, because of rate limits, it's best to test this method for one candidate before trying
    on all candidates.

    Args:
        twitter.......The TwitterAPI object
        screen_name... a string of a Twitter screen name
    Returns:
        A list of ints, one per friend ID, sorted in ascending order.

    Note: If a user follows more than 5000 accounts, we will limit ourselves to
    the first 5000 accounts returned.

    In this test case, I return the first 5 accounts that I follow.
    >>> twitter = get_twitter()
    >>> get_friends(twitter, 'aronwc')[:5]
    [695023, 1697081, 8381682, 10204352, 11669522]
    """
    ###TODO
    request = robust_request(twitter,'friends/ids',{'screen_name': screen_name })
    requestUsersList = request.json()
    return sorted(requestUsersList['ids'][:2000])


def add_all_friends(twitter, users):
    """ Get the list of accounts each user follows.
    I.e., call the get_friends method for all 4 candidates.

    Store the result in each user's dict using a new key called 'friends'.

    Args:
        twitter...The TwitterAPI object.
        users.....The list of user dicts.
    Returns:
        Nothing

    >>> twitter = get_twitter()
    >>> users = [{'screen_name': 'aronwc'}]
    >>> add_all_friends(twitter, users)
    >>> users[0]['friends'][:5]
    [695023, 1697081, 8381682, 10204352, 11669522]
    """
    ###TODO
    updatedUsers = []
    for user in users:
        temp_screen_name = user['screen_name']
        user['friends'] = get_friends(twitter,temp_screen_name)
        updatedUsers.append(user)


def print_num_friends(users):
    """Print the number of friends per candidate, sorted by candidate name.
    See Log.txt for an example.
    Args:
        users....The list of user dicts.
    Returns:
        Nothing
    """
    ###TODO
    for user in users:
        print(user['screen_name'],' ',user['friends_count'])
    pass


def count_friends(users):
    """ Count how often each friend is followed.
    Args:
        users: a list of user dicts
    Returns:
        a Counter object mapping each friend to the number of candidates who follow them.
        Counter documentation: https://docs.python.org/dev/library/collections.html#collections.Counter

    In this example, friend '2' is followed by three different users.
    >>> c = count_friends([{'friends': [1,2]}, {'friends': [2,3]}, {'friends': [2,3]}])
    >>> c.most_common()
    [(2, 3), (3, 2), (1, 1)]
    """
    ###TODO
    friendsList = []
    for user in users:
        friendsList += sorted(user['friends'])
    c = Counter(sorted(friendsList))      
    return c

def save_users_details_to_file(users, path):
    for u in users:
        json.dump(u, open(path + u['screen_name'] + ".txt", 'w'))
    #Saving Users details to pickle
    pickle.dump(users, open(path + 'users.pkl','wb'))


def create_graph(users, friend_counts):
    """ Create a networkx undirected Graph, adding each candidate and friend
        as a node.  Note: while all candidates should be added to the graph,
        only add friends to the graph if they are followed by more than one
        candidate. (This is to reduce clutter.)

        Each candidate in the Graph will be represented by their screen_name,
        while each friend will be represented by their user id.

    Args:
      users...........The list of user dicts.
      friend_counts...The Counter dict mapping each friend to the number of candidates that follow them.
    Returns:
      A networkx Graph
    """
###TODO
    friend_counts = {}
    for user in users:
        friend_counts[user['screen_name']] =  sorted(user['friends'])
    friendsList = []
    for user in users:
        friendsList += sorted(friend_counts.get(user['screen_name']))
    c = Counter(sorted(friendsList))
    graph = nx.Graph()
    for user in users:
        # Add a node
        graph.add_node(user['screen_name'])
        for cnt in range(len(friend_counts.get(user['screen_name']))):
            if(c[friend_counts.get(user['screen_name'])[cnt]] > 1):
                graph.add_node(friend_counts.get(user['screen_name'])[cnt])
                graph.add_edge(user['screen_name'],friend_counts.get(user['screen_name'])[cnt])
                
    return graph

def draw_network(graph, users, filename):
    """
    Draw the network to a file. Only label the candidate nodes; the friend
    nodes should have no labels (to reduce clutter).

    Methods you'll need include networkx.draw_networkx, plt.figure, and plt.savefig.

    Your figure does not have to look exactly the same as mine, but try to
    make it look presentable.
    """
    ###TODO
    # Draw the graph
    labels = {}
    for user in users:
        labels[user['screen_name']] = user['screen_name']
    pos = nx.spring_layout(graph)
    plt.gcf()
    plt.figure(3,figsize=(12,12)) 
    plt.axis('off')
    nx.draw_networkx_nodes(graph,pos,alpha=0.6,node_size=50,labels=True)
    nx.draw_networkx_edges(graph,pos,alpha=0.7,width=0.1)
    nx.draw_networkx_labels(graph,pos,labels,font_size=10,font_color='black')
    plt.savefig(filename, format="PNG")


def fetch_tweets(twitter, users, path):
    """ 
    This method is used for fetching tweets for the available users. 
    Restriction for fetching tweets is 100 tweets per user.

    Args:
      twitter.........Twitter objects containing all connection details.
      users...........The list of user dicts.
      path............Path where twitter details will be saved
    Returns:
      Tuple of tweets for training and testing
    """

    tweets_data=[]
    tweets_collection_by_users = dict()
    tweets_collection_for_testing_by_user = dict()
    for user in users:
        tweets_data = []
        params = {'screen_name': user['screen_name'], 'include_rts': False, 'count': 200} #, 'page':i}
        user_detailed_tweets = robust_request(twitter, 'statuses/user_timeline', params)
        response = [tweet['text'] for tweet in user_detailed_tweets]
        tweets_data += response
        tweets_collection_by_users[user['screen_name']] = tweets_data[:50]
        tweets_collection_for_testing_by_user[user['screen_name']] = tweets_data[50:100]
        
        json.dump(tweets_collection_by_users[user['screen_name']], open(path + 'train/' + user['screen_name'] + '_tweets.txt', 'w'))
        pickle.dump(tweets_collection_by_users, open(path + 'train/' + 'tweets.pkl','wb'))
        
        json.dump(tweets_collection_for_testing_by_user[user['screen_name']], open(path + 'test/' + user['screen_name'] + '_tweets.txt', 'w'))
        pickle.dump(tweets_collection_for_testing_by_user, open(path +  'test/' + 'tweets.pkl','wb'))
    return tweets_collection_by_users,tweets_collection_for_testing_by_user


def write_tweets_to_excel(tweets,users,path):
    """ 
    This method is used for saving fetched tweets to excel files

    Args:
      tweets..........Contains Tweets dictionary based on the users screen name
      users...........The list of user dicts.
      path............Path where excel files will be saved
    """
    # Create a workbook and add a worksheet.
    workbook = xlsxwriter.Workbook(path + 'tweets.xlsx')
    worksheet = workbook.add_worksheet()

    # Start from the first cell. Rows and columns are zero indexed.
    row = 0
    col = 0

    for user in users:
        tweets_by_user = tweets[user['screen_name']]
        for tweet in tweets_by_user:
            # Col for Screen Name
            worksheet.write(row, col,     user['screen_name'])
            # Col for Tweets text
            worksheet.write(row, col + 1, tweet)
            # Col for sentiments - passed blank and will be done manually
            worksheet.write(row, col + 2, '')
            row += 1

    workbook.close()

def main():
    twitter = get_twitter()
    screen_names = read_screen_names('candidates.txt')
    print('Established Twitter connection.')
    print('Read screen names: %s' % screen_names)

    users = sorted(get_users(twitter, screen_names), key=lambda x: x['screen_name'])
    print('found %d users with screen_names %s' %
        (len(users), str([u['screen_name'] for u in users])))
    add_all_friends(twitter, users)
    print('Friends per candidate:')
    print_num_friends(users)

    friend_counts = count_friends(users)
    print('Most common friends:\n%s' % str(sorted(sorted(friend_counts.most_common(5)),key=lambda a: a[1],reverse=True)))
    path_user_details = "./users/"
    print('Saving User Details to file inside users folder.')
    save_users_details_to_file(users,path_user_details)

    graph = create_graph(users, friend_counts)
    print('Saving Graph object to pickle file.')
    pickle.dump(graph, open('./graph.pkl','wb'))
    print('graph has %s nodes and %s edges' % (len(graph.nodes()), len(graph.edges())))
    draw_network(graph, users, './network.png')
    print('network drawn to network_new.png')

    print('Fetching and Saving Users Tweets dictionary to text file and pickle file in tweets folder.')
    tweets_train, tweets_test = fetch_tweets(twitter, users,'./tweets/')

    print("Saving Training and Testing data to excel file")
    write_tweets_to_excel(tweets_train, users, './tweets/train/')
    write_tweets_to_excel(tweets_test, users, './tweets/test/')     

if __name__ == "__main__":
    main()