"""
Summarize data.
"""
import pickle

"""
This method is used for fetching the user details from pickle file.
"""
def load_users_from_file(path, file_name):
    users = pickle.load(open(path + file_name,'rb'))
    return users

"""
This method is used for fetching tweets information from pickle file.
"""
def load_tweets_from_file(path):
    tweets = pickle.load(open(path + 'tweets.pkl','rb'))
    return tweets

"""
This method is used for fetching cluster information from pickle file.
"""
def load_clusters_from_file(path):
    clusters = pickle.load(open(path + 'clusters.pkl','rb'))
    return clusters

"""
This method is used for fetching misclassified document information from pickle file.
"""
def load_misclassified_list_from_file(path):
    misclassified_list = pickle.load(open(path + 'misclassified_list.pkl','rb'))
    return misclassified_list

"""
This method is used for fetching positive document information from pickle file.
"""
def load_positive_predicted_list_from_file(path):
    positive_predicted_list = pickle.load(open(path + 'positive_predicted_list.pkl','rb'))
    return positive_predicted_list

"""
This method is used for fetching negative document information from pickle file.
"""
def load_negative_predicted_list_from_file(path):
    negative_predicted_list = pickle.load(open(path + 'negative_predicted_list.pkl','rb'))
    return negative_predicted_list

"""
This method is used for fetching accuracy information from pickle file.
"""
def load_test_accuracy_from_file(path):
    test_accuracy = pickle.load(open(path + 'test_accuracy.pkl','rb'))
    return test_accuracy

"""
summarizing the User information and details
"""
def summarize_users(users, file):
    file.write("\n------------------------------------------------\n")
    file.write("\nNumber of users collected :-\n")
    file.write("____________________________\n")
    for user in users:
        file.write(user['screen_name']+ "\n")
    file.write("\nNo of friends per user:-\n")
    file.write("________________________\n")
    for user in users:
        file.write(str(user['screen_name']) + ' \t' + str(user['friends_count'])+"\n")
    file.write("------------------------------------------------\n")

"""
summarizing Tweets details
"""
def summarize_messages(train, test, users, file):
    file.write("\n------------------------------------------------\n")
    file.write("\nNumber of tweets collected for training:-\n")
    file.write("___________________________________________\n")
    for user in users:
        file.write(str(user['screen_name']) + ' \t' + str(len(train[user['screen_name']]))+"\n")
    
    file.write("\nNumber of tweets collected for testing:-\n")
    file.write("___________________________________________\n")
    for user in users:
        file.write(str(user['screen_name']) + ' \t' + str(len(test[user['screen_name']]))+"\n")
    file.write("------------------------------------------------\n")

"""
summarizing communities details
"""
def summarize_clusters(clusters,file):
    file.write("------------------------------------------------\n")
    no_of_clusters = 0
    total_no_of_users = 0
    for c in clusters:
        if( c.order() > 1):
            no_of_clusters += 1
            total_no_of_users += c.order()
    file.write("Number of communities discovered length >1:\t" + str(no_of_clusters)+"\n")
    file.write("Average number of users per community:\t" + str(total_no_of_users/no_of_clusters)+"\n")
    file.write("\nUsers per communities:-\n")
    file.write("_________________________\n")
    for c in clusters:
        if( c.order() > 1):
            file.write("\n"+str(c.nodes())+"\n")
    file.write("------------------------------------------------\n")

"""
summarizing training and prediction details
"""
def summarize_predictions(file):
    file.write("------------------------------------------------\n")
    file.write("\n Number of instances per class found:\n")
    file.write("_______________________________________\n")
    misclassified_list = load_misclassified_list_from_file('./')
    positive_predicted_list = load_positive_predicted_list_from_file('./')
    negative_predicted_list = load_negative_predicted_list_from_file('./')
    
    file.write("\nInstances for positive class:\t\t" + str(len(positive_predicted_list))+"\n")
    file.write("\nInstances for negative class:\t\t" + str(len(negative_predicted_list))+"\n")
    file.write("\nInstances for misclassified class:\t" + str(len(misclassified_list))+"\n\n")
    
    file.write("Overall Testing Accuracy = \t" + str(load_test_accuracy_from_file('./')) + "\n")
    
    file.write("\n\n********Example from each class********\n")
    
    file.write("\n\nTOP POSITIVE TEST DOCUMENTS:\n")
    file.write("________________________________\n\n")
    for document in sorted(list(positive_predicted_list.items()), key=lambda x:-x[1]['proba'])[:5]:
        file.write(('truth=%s predicted=%s proba=%.6f' %  (str(document[1]["truth"]),str(document[1]["predicted"]),document[1]["proba"])))
        file.write("\n"+(str(document[1]["doc"]) + "\n\n"))
    
    file.write("\n\nTOP NEGATIVE TEST DOCUMENTS:\n")
    file.write("________________________________\n\n")
    for document in sorted(list(negative_predicted_list.items()), key=lambda x:-x[1]['proba'])[:5]:
        file.write(('truth=%s predicted=%s proba=%.6f' %  (str(document[1]["truth"]),str(document[1]["predicted"]),document[1]["proba"])))
        file.write("\n"+(str(document[1]["doc"]) + "\n\n"))
                
    file.write("\n\nTOP MISCLASSIFIED TEST DOCUMENTS:\n")
    file.write("_____________________________________\n\n")
    for document in sorted(list(misclassified_list.items()), key=lambda x:-x[1]['proba'])[:5]:
        file.write(('truth=%s predicted=%s proba=%.6f' %  (str(document[1]["truth"]),str(document[1]["predicted"]),document[1]["proba"])))
        file.write("\n"+(str(document[1]["doc"]) + "\n\n"))
    
    
    file.write("------------------------------------------------\n")
    
def main():
    # Creating Summary.txt file for writting
    file = open("summary.txt","w+") 
    # Loading User Details
    users = load_users_from_file('./users/','users.pkl')
    #Summarizing User Details
    summarize_users(users, file)
    # Loading Tweets for training and Testing
    tweets_train = load_tweets_from_file('./tweets/train/')
    tweets_test = load_tweets_from_file('./tweets/test/')
    # Summarizing Tweets
    summarize_messages(tweets_train,tweets_test,users,file)
    # Loading clusters information
    clusters = load_clusters_from_file('./')
    # Summarizing clusters 
    summarize_clusters(clusters,file)
    # summarizing predictions and accuracy
    summarize_predictions(file)
    file.close()
    print("All the results written into summary.txt")

if __name__ == "__main__":
    main()