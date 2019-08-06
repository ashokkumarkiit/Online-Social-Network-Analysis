import networkx as nx

def example_graph():
    """
    For the testing purpose, I have used the sample graph that
    was given in the Assignment 1
    """
    g = nx.Graph()
    g.add_edges_from([('A', 'B'), ('A', 'C'), ('B', 'C'), ('B', 'D'), ('D', 'E'), ('D', 'F'), ('D', 'G'), ('E', 'F'), ('G', 'F')])
    return g

def jaccard_wt(graph, node):
    """
    The weighted jaccard score, defined above.
    Args:
      graph....a networkx graph
      node.....a node to score potential new edges for.
    Returns:
      A list of ((node, ni), score) tuples, representing the 
                score assigned to edge (node, ni)
                (note the edge order)
    """
    #Fetching neighbours of the given node
    neighbors = set(graph.neighbors(node))
    scores = []
    
    # As the dinominator part will be same for all the combination
    # Calculating in the beginning
    denominator_1 = 0
    for s_node in neighbors:
        denominator_1 += graph.degree(s_node)
    
    # Looping for all the graph nodes
    for n in graph.nodes():
        
        #Removing the node itself and others the already have an edge with given node
        if(n not in graph.neighbors(node) and n != node):
            neighbors2 = set(graph.neighbors(n))
            
            # Taking out the common neighbours 
            combined_neig = set(neighbors & neighbors2)
            numerator = 0
            denominator_2 = 0
            if(len(combined_neig) > 0):
                for s_node in combined_neig:
                    numerator += 1/graph.degree(s_node)
            
            for s_node in neighbors2:
                denominator_2 += graph.degree(s_node)
            
            # Calculating the jaccard Similarity with the new given formula
            jac_score = numerator / (1/denominator_1 + 1/denominator_2 )
            scores.append(((node,n), jac_score))
    return sorted(scores, key=lambda x: (-x[1], x[0][1]))

def main():
    graph = example_graph()
    for node in graph.nodes():
        print("Potential new edges for node = " , node, "\n" ,jaccard_wt(graph,node),"\n")

if __name__ == '__main__':
    main()