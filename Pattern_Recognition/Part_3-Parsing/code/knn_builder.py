'''
@author: Sanyukta Kate, Pratik Bongale
'''

import math
from operator import itemgetter
import symbol_features

def get_graph(ink, k):
    '''
    Gets a KNN graph with provided k. distance between bounding box centers of symbols is used to determine nearest neighbors
    :param ink: Inkml object containing segments/symbols
    :param k: number of nearest neighbors to consider.
    :return:
    '''

    segments = ink.segments
    seg_material = symbol_features.build_seg_materials(segments, ink)

    if k>=len(seg_material[0]):
        k = len(seg_material[0])-1

    #seg_id_ref = create_segment_refs()
    distance_matrix = find_distance_matrix(seg_material[0])
    G = build_KNN_Graph(distance_matrix, seg_material[0], k)

    return G

def build_KNN_Graph(distance_matrix, seg_map, k):
    # G graph: dictionary, G = {'s1': [s2,s3,s4], 's2':[s1,s3,s4]}; where s1, s2, etc are segment ids of the expression
    G = {}

    seg_id_ref = create_segment_refs(seg_map)

    # Get the symbol ids
    seg_ids = seg_map.keys()

    # create a empty neighbors list for G symbols
    for symbol in seg_ids:
        G[symbol] = list()
        #print(type(symbol))
    #print(G)

    for i in range(0, len(seg_ids)):
        temp = {}
        for j in range(0, len(seg_ids)):
            if i!=j:
                temp[j] = distance_matrix[i][j]

        # sort the temp dictionary on basis of value
        sorted_temp = sorted(temp.items(), key = itemgetter(1))

        # Take the first k values from the sorted list which will be the neighbors
        neighbors = []
        for value in range(0, k): # find the k neighbors
            #neighbors.append(seg_id_ref[sorted_temp[value][0]]) # append the neighbors into the list
            #print("abc",G[seg_id_ref[i]])
            if seg_id_ref[sorted_temp[value][0]] not in G[seg_id_ref[i]]:
                # add it to the list
                G[seg_id_ref[i]].append(seg_id_ref[sorted_temp[value][0]])
                # get the neighbors of seg_id_ref[sorted_temp[value][0]]
            other_sym_neighs = G[seg_id_ref[sorted_temp[value][0]]]

            # check if the current symb is present in the other_neighs, if not, add it
            if seg_id_ref[i] not in other_sym_neighs:

                # add it in other neighbors list
                G[seg_id_ref[sorted_temp[value][0]]].append(seg_id_ref[i])

        #G[seg_id_ref[i]] = neighbors # For symbol i, the neighbor is going to the list of neighbors
    #print(G)
    return G

def create_segment_refs(seg_map):
    # create_segment_refs creates segments in
    # required for parsing the distance matrix and recognizing the symbols

    seg_ids = seg_map.keys() # get the symbol ids
    seg_id_ref = {}

    index = 0
    for key in seg_ids:
        seg_id_ref[index] = key   # 0 is mapped to the first symbol
        index += 1

    return seg_id_ref

def find_distance_matrix(seg_map):
    # 1. Get all the keys from seg_map which will have the semgnet ids
    # 2. Create another dictionary....reference segments, seg_id_ref = {"index" = seg_id}
    # 3. Create a multi - dimensional list of N*N size which will store all the distances of the symbols from each other
    # 4. Now, take the first symbol from the key, and find its distance from s2,
    #    store the (s1 s2) distance in the matrix, by accessing it's indices from the seg_id_ref dictionary.

    # get all the keys, which is the segment ids or the symbol ids
    seg_ids = seg_map.keys()
    no_segments = len(seg_ids)

    # create a seg_id_ref dictionary
    seg_id_ref = create_segment_refs(seg_map)

    # Create a matrix (N*N)
    distance_matrix = [[0 for i in range(no_segments)] for i in range(no_segments)]

    s1 = 0
    while s1<no_segments:
        s2 = s1
        while s2<no_segments:
            # find the distance of s1 and s2
            if s1!=s2:
                # find the euclidean distance
                d = find_distance(seg_map[seg_id_ref[s1]],seg_map[seg_id_ref[s2]])

                # put this distance d in the matrix
                distance_matrix[s1][s2] = d
                distance_matrix[s2][s1] = d
            s2+=1
        s1+=1

    return distance_matrix

def find_distance(s1_bb_center, s2_bb_center):

    # find the distance between s1 and s2 symbols

    distance = math.hypot(s2_bb_center[0] - s1_bb_center[0], s2_bb_center[1] - s1_bb_center[1])
    return distance

if __name__ == "__main__":
    seg_map = {"25": [15, 10], "26":[20, 10], "24": [10, 10], "27":[30,10]}
    get_graph(seg_map, 2)
