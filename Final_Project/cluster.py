"""
Cluster data.
"""
import pickle
import networkx as nx

def load_graph_from_file():
    """
    This method loads the graph from pickle file.
    Returns:
        Graph object containing graph 
    """
    graph = pickle.load(open('./graph.pkl','rb'))
    return graph

def get_subgraph(graph, min_degree):
    """Return a subgraph containing nodes whose degree is
    greater than or equal to min_degree.
    We'll use this in the main method to prune the original graph.

    Params:
      graph........a networkx graph
      min_degree...degree threshold
    Returns:
      a networkx graph, filtered as defined above.

    >>> subgraph = get_subgraph(example_graph(), 3)
    >>> sorted(subgraph.nodes())
    ['B', 'D', 'F']
    >>> len(subgraph.edges())
    2
    """
    ###TODO
    lst = []
    for node in graph.nodes():
        if(len(sorted(graph.neighbors(node))) >= min_degree ):
            lst.append(node)

    subgraph = graph.subgraph(lst)
    return(subgraph)


def edges_to_remove(G):
    """
    Method used for calculating the betweeness of each edge of the graph
    
    params:
        G....... Contains Graph that is used for computing betweeness
    Return:
        Returns the sorted list of betweeness in descending order with edges as tuples
    """
    dict1 = nx.edge_betweenness_centrality(G, normalized=False) #
    list_of_tuples = dict1.items()
    temp = sorted(list_of_tuples, key = lambda x:x[1], reverse = True)
    temp1 = [(x[0],round(x[1], 2)) for x in temp]
    return temp1 #[0][0]

def girvan(Gr):
    """
    Algorithm used for identifying clusters in a given graph.
    The clusters are calculated until the the specified no of communities are identified.

    params:
        G...... Graph used in algorithm
    Returns:
        The cluster containing all the clusters that have been identified.
    """
    G = Gr.copy()
    clusters = [c for c in nx.connected_component_subgraphs(G)]
    no_of_clusters = len(clusters)
    edges = edges_to_remove(G)
    index = 0
    max_bt = edges[index][1]
    while(no_of_clusters < 5):
        if(max_bt == edges[index][1]):
            G.remove_edge(*edges[index][0])
            clusters = [c for c in nx.connected_component_subgraphs(G)]
            no_of_clusters = len(clusters)
            index +=1
        else:
            edges = edges_to_remove(G)
            max_bt = edges[index][1]
            index = 0
    return clusters

def main():
    graph = load_graph_from_file()
    print('graph has %d nodes and %d edges' %(graph.order(), graph.number_of_edges()))
    subgraph = get_subgraph(graph, 2)
    print('subgraph has %d nodes and %d edges' %
        (subgraph.order(), subgraph.number_of_edges()))
    print("Clustering process initiated...")
    clusters = girvan(subgraph)
    print("Clustering process finished...")
    print('%d clusters' % len(clusters))
    clusters = sorted(clusters, key=lambda x:len(x.nodes), reverse=True)

    # Dumps cluster information to pickle file
    pickle.dump(clusters, open('./clusters.pkl','wb'))
    no_of_clusters = 0
    total_no_of_users = 0
    for c in clusters:
        if( c.order() > 1):
            no_of_clusters += 1
            total_no_of_users += c.order()
    print("Number of communities discovered with size more than one :", no_of_clusters)
    print("Average number of users per community:", total_no_of_users/no_of_clusters)

    print("\nUsers per communities:-")
    for c in clusters:
        if( c.order() > 1):
            print("\n"+str(c.nodes())+"\n")

if __name__ == "__main__":
    main()