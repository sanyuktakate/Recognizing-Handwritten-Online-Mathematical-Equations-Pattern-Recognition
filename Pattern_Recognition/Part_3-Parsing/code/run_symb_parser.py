'''
@author: Sanyukta Kate, Pratik Bongale
'''

import pickle
import gzip
import ntpath
import sys
import numpy as np
from data_preparation import preprocess
from helper import parse_raw_ink_data, find_leftmost_symb, make_sc
from inkml import Inkml
import os
import knn_builder
from collections import defaultdict
import edmonds
from symbol_features import *

def parser_test(par_model_pkl, out_dname, test_file_dir, test_files):
    '''
    Test a symbol level spatial relationship parser model on ground truth symbols
    :param par_model_pkl: pickle filename of trained parser model
    :param out_dname: directory name for output lg files(directory will be created if it does'nt exist)
    :param test_files_dir: directory path of .inkml files to test
    :param test_files: a file listing all inkml filenames on which parser will be tested
    :return:
    '''

    # load the symbol relationship classifier
    with gzip.open(par_model_pkl, 'rb') as parser_pickle:
        parser = pickle.load(parser_pickle)

    # read all the files in the current directory and sub directories recursively and built a dictionary
    file_dictionary = dict()
    for (dirpath, dirnames, filenames) in os.walk(test_file_dir):
        for file in filenames:
            file_dictionary[file] = os.path.join(dirpath, file)

    expressions = list()  # list of inkml objects
    # check if only a single file is given as input
    if test_files.endswith(".inkml"):
        abs_file_path = file_dictionary[test_files]

        try:
            file_obj = Inkml(abs_file_path)

            # process stroke points str -> 2d np-array
            parse_raw_ink_data(file_obj)

            # perform preprocessing (smoothing, duplicate removal, resampling)
            preprocess(file_obj)

            expressions.append(file_obj)
        except:
            print('error reading file: ' + test_files)
            return

    else:
        # convert each file in test data to an inkml object
        with open(test_files, 'r') as test_files:

            for fname in test_files:

                fname = fname.strip()

                if fname.endswith('.inkml'):

                    abs_file_path = file_dictionary[fname]

                    try:
                        # create Inkml object
                        file_obj = Inkml(abs_file_path)
                    except:
                        print('error occured in file: ' + fname)
                        continue

                    # process stroke points str -> 2d np-array
                    parse_raw_ink_data(file_obj)

                    # perform preprocessing (smoothing, duplicate removal, resampling)
                    preprocess(file_obj)

                    expressions.append(file_obj)
        print('Preprocessing complete')

    # For each expression
    # 1. Find the KNN graph
    # 2. Take symbol pairs from the KNN graph and then find the features of the symbol pair
    # 3. Predict relationships using parser
    # 4. Build a weighted symbol level fully connected graph
    # 5. Find parse tree using Edmond's algorithm

    # print('Processing file:')
    for ink in expressions:

        # print(ink.fileName)

        # handle the case where there is just one symbol(no relationships)
        if len(ink.segments) <= 1:
            write_to_lg(ink, [], [], out_dname)
            continue

        knn_graph = knn_builder.get_graph(ink, k=6)   # root is leftmost symbol in expr

        if not edmonds.is_sc(knn_graph):
            knn_graph = make_sc(knn_graph)

        edm_graph = dict()
        rel_graph = dict()
        for u in knn_graph:
            for v in knn_graph[u]:
                f_vector = get_symb_features(u, v, ink)
                rel_prob = parser.predict_proba(f_vector)     # get a probability vector for all relationships
                idx = np.argsort(-rel_prob)                     # sort probabilities in descending order
                pred_cls = parser.classes_[idx]                 # find the relationship labels

                if u in edm_graph:
                    edm_graph[u][v] = max(rel_prob[0])
                else:
                    edm_graph[u] = { v : max(rel_prob[0]) }

                if u in rel_graph:
                    rel_graph[u][v] = pred_cls[0][0]
                else:
                    rel_graph[u] = { v : pred_cls[0][0] }


        root = find_leftmost_symb(ink)      # get the segment id of leftmost symbol

        # negate the weights to get a maximum spanning tree
        edmonds.negate_wts(edm_graph)

        mst_edge_lst = edmonds.get_mst(edm_graph, root)     # returns directed edges forming MST

        write_to_lg(ink, mst_edge_lst, rel_graph, out_dname)

    print('Output label graph(.lg) files can be found in directory: output_lg_' + out_dname)

def write_to_lg(ink, edges, rel, out_dir_name):
    '''
    writes objects stored in ink, and relationships between edges in mst to .lg file
    :param ink: an instance of Inkml class storing strokes and segments representing symbols in expression
    :param edges: a list of edges forming a directed maximum spanning tree
    :param rel: dictionary representing a graph with edges labeled with relationships
    :return:
    '''

    lines = list()
    lines.append("# IUD, %s\n" % ink.UI)
    lines.append("# Objects(%d):\n" % len(ink.segments))

    # build segMap {obj_id : seg_id}
    object_count = defaultdict(lambda: 0)  # stores the count of each object
    for s in ink.segments:
        seg = ink.segments[s]
        if seg.label.startswith(","):
            seg.label = "COMMA" + str(seg.label[1:])
        object_count[seg.label] += 1

    for s in ink.segments:
        seg = ink.segments[s]
        # formulate a object_id for each object
        objId = "{}_{}".format(seg.label.strip("\\"), object_count[seg.label])
        object_count[seg.label] -= 1
        ink.segMap[objId] = s

    # write objects to lg file
    rev_segMap = dict()
    for symb_id, seg_id in ink.segMap.items():
        label = ink.segments[seg_id].label.strip()

        strokes = ink.segments[seg_id].strId
        rev_segMap[seg_id] = symb_id

        row = ['O', symb_id, label, '1.0']
        row.extend(strokes)

        formatted_row = ""
        for i, ele in enumerate(row):
            formatted_row += ele + (", " if i < (len(row) - 1) else "\n")
        lines.append(formatted_row)

    # add relations to lg file
    lines.append("\n# Relations from SRT:\n")
    if edges and rel:
        for (u, v) in edges:    # edges are stored in form of tuples (start_seg_id, end_seg_id)
            label = rel[u][v]       # get the relationship label for edge e
            row = ['R', rev_segMap[u], rev_segMap[v], label, '1.0']

            formatted_row = ""
            for i, ele in enumerate(row):
                formatted_row += ele + (", " if i < (len(row) - 1) else "\n")
            lines.append(formatted_row)

    out_fname = str(ntpath.basename(ink.fileName).split(".")[0]) + ".lg"

    # create a directory to store the output .lg files
    output_dir = 'out_lg_' + out_dir_name
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(os.path.join(output_dir, out_fname), 'w') as out_file:
        out_file.writelines(lines)

if __name__ == '__main__':

    if len(sys.argv) != 5:
        # model_pickle, tr_gt, dir_path, symbols_to_test
        print('Usage: \npython run_parser.py <parser_pkl> <out_dir_name> <abs_dir_path> <test_expressions>\n')
        sys.exit(0)

    parser_model = sys.argv[1]
    out_dname = sys.argv[2]
    abs_dir_path = sys.argv[3]
    symbols_to_test = sys.argv[4]

    parser_test(parser_model, out_dname, abs_dir_path, symbols_to_test)

    '''
    rf_parser.pklz
    testset
    /home/pratik/Desktop/PR/Projects/p3/dataset/Inkml_lg/TrainINKML
    test_files.txt
    '''