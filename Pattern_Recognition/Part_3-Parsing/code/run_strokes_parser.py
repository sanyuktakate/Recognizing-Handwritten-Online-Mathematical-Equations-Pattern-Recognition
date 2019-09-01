'''
@author: Sanyukta Kate, Pratik Bongale
'''

import pickle
import gzip
import ntpath
import sys
from collections import defaultdict
import numpy as np
import feature_generation
import preprocessing
import shape_context_features
from data_preparation import preprocess
from helper import parse_raw_ink_data, find_leftmost_symb, make_sc
from inkml import Inkml, Segment
import os
import knn_builder
import edmonds
from symbol_features import *

def parser_test(parser_pkl, segmenter_pkl, symb_clf_pkl, out_dname, abs_dir_path, test_data):
    '''
    Test a symbol level spatial relationship parser model on ground truth symbols
    :param parser_pkl: pickle filename of trained parser model
    :param segmenter_pkl: pickle filename of trained parser model
    :param symb_clf_pkl: pickle filename of trained parser model
    :param out_dname: directory name for output lg files(directory will be created if it does'nt exist)
    :param test_files_dir: directory path of .inkml files to test
    :param test_files: a file listing all inkml filenames on which parser will be tested
    :return:
    '''

    # load the symbol classifier
    with gzip.open(symb_clf_pkl, 'rb') as symb_clf_pickle:
        symb_clf = pickle.load(symb_clf_pickle)

    # load the segmenter (merge/split classifier)
    with gzip.open(segmenter_pkl, 'rb') as seg_pickle:
        segmenter = pickle.load(seg_pickle)

    # load the symbol relationship classifier
    with gzip.open(parser_pkl, 'rb') as parser_pickle:
        parser = pickle.load(parser_pickle)

    # read all files in current directory and sub directories recursively and built a dictionary
    file_dictionary = dict()
    for (dirpath, dirnames, filenames) in os.walk(abs_dir_path):
        for file in filenames:
            file_dictionary[file] = os.path.join(dirpath, file)

    expressions = list()  # list of inkml objects

    # check if only a single file is given as input
    if test_data.endswith(".inkml"):
        abs_file_path = file_dictionary[test_data]

        try:
            file_obj = Inkml(abs_file_path)

            # process stroke points str -> 2d np-array
            parse_raw_ink_data(file_obj)

            # perform preprocessing (smoothing, duplicate removal, resampling)
            preprocess(file_obj)

            expressions.append(file_obj)
        except:
            print('error reading file: ' + test_data)
            return

    else:
        # convert each file in test data to an inkml object
        with open(test_data, 'r') as test_files:
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

                    # perform preprocessing (smoothing, normalization, duplicate removal, resampling)
                    preprocess(file_obj)

                    expressions.append(file_obj)

    print('preprocessing complete')

    # For each expression
    # Get segments using segmenter(Build a time series graph, check with Merge/Split classifier)
    # Find the KNN graph
    # Take symbol pairs from the KNN graph and then find the features of the symbol pair
    # Predict relationships using parser
    # Build a weighted symbol level fully connected graph
    # Find parse tree using Edmond's algorithm

    for ink in expressions:

        segments = get_segments(ink, segmenter, symb_clf)

        ink.segments = {}  # empty the existing segments read from inkml(in any)

        # add the segments detected
        for s in segments:
            ink.segments[s.id] = s

        # handle the case where there is just one symbol(no relationships)
        if len(ink.segments) <= 1:
            write_to_lg(ink, [], [], out_dname)
            continue

        knn_graph = knn_builder.get_graph(ink, k=6)  # root is leftmost symbol in expr

        if not edmonds.is_sc(knn_graph):
            knn_graph = make_sc(knn_graph)

        edm_graph = dict()
        rel_graph = dict()
        for u in knn_graph:
            for v in knn_graph[u]:
                f_vector = get_symb_features(u, v, ink)
                rel_prob = parser.predict_proba(f_vector)  # get a probability vector for all relationships
                idx = np.argsort(-rel_prob)  # sort probabilities in descending order
                pred_cls = parser.classes_[idx]  # find the relationship labels

                if u in edm_graph:
                    edm_graph[u][v] = max(rel_prob[0])
                else:
                    edm_graph[u] = {v: max(rel_prob[0])}

                if u in rel_graph:
                    rel_graph[u][v] = pred_cls[0][0]
                else:
                    rel_graph[u] = {v: pred_cls[0][0]}


        root = find_leftmost_symb(ink)  # get the segment id of leftmost symbol

        # negate the weights to get a maximum spanning tree
        edmonds.negate_wts(edm_graph)

        mst_edge_lst = edmonds.get_mst(edm_graph, root)  # returns directed edges forming MST

        # print('Writing to lg file')
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

def get_segments(ink, seg_clf, symb_clf):
    nStrokes = len(ink.strokes)

    # if we have only one stroke in the expression, send the value to final classifier
    if nStrokes <= 1:
        seg_id = '2'
        strID = list(ink.strokes.keys())  # will only have a single stroke
        label = get_symbol_label(symb_clf, [ink.strokes[strID[0]]])
        segment = Segment(seg_id, label, set(strID))  # create a segment object for this symbol
        return [segment]

    # find whether to merge or split successive strokes
    merged_strokes = list()
    for i in range(nStrokes - 1):
        s1 = ink.strkOrder[i]  # stroke id 1
        s2 = ink.strkOrder[i + 1]

        geo_feature_vec = geometric_features.get_pair_features(ink.strokes[s1], ink.strokes[s2])
        shape_feature_vec = shape_context_features.shape_features(ink.strokes[s1], ink.strokes[s2])

        feature_vec = geo_feature_vec + shape_feature_vec

        # predict using binary classifier
        if seg_clf.predict([feature_vec]) == [1]:  # merge
            merged_strokes.append({s1, s2})

    # find connected components
    res_symbols = list()

    # while merge list is not empty
    while merged_strokes:

        # find all the connected consecutive stroke pairs
        n = len(merged_strokes)
        temp = set()
        i = 1

        while i < n and merged_strokes[i - 1].intersection(merged_strokes[i]):
            temp.update(merged_strokes[i - 1])
            i += 1

        temp.update(merged_strokes[i - 1])
        res_symbols.append(temp)
        del merged_strokes[:i]

    # find the best segmentation candidate
    all_strokes = set(ink.strkOrder)

    # remove strokes which are already merged
    for cc_strokes in res_symbols:
        all_strokes = all_strokes - cc_strokes

    # all remaining strokes form one symbol each
    for s in all_strokes:
        res_symbols.append({s})

    # predict class label for each symbol
    seg_id = max([int(a) for a in ink.strkOrder]) + 1
    segments = list()
    for i, symb_strokes in enumerate(res_symbols):
        seg_id += i
        strID = symb_strokes
        label = get_symbol_label(symb_clf, [ink.strokes[s] for s in symb_strokes])

        # create a segment object for this symbol
        segments.append(Segment(str(seg_id), label, strID))

    return segments

def get_symbol_label(clf, strokes):
    '''
    Gets the label as predicted by the symbol classifier
    :param clf: classifier model
    :param strokes: a list of strokes representing the symbol
    :return: predicted label
    '''
    strokes_lst = list()

    for s in strokes:
        strokes_lst.append({"x": s[:, 0], "y": s[:, 1]})

    # get symbol features
    preprocessing.resample(strokes_lst)
    feature_vec = feature_generation.gen_feature_vector(strokes_lst)

    # possible class values
    # class_lst = clf.classes_
    # print(class_lst)

    cls = clf.predict([feature_vec])

    # handling predicted comma's
    if cls[0] == ",":
        cls[0] = "COMMA"

    return cls[0]

if __name__ == '__main__':

    if len(sys.argv) != 7:
        # model_pickle, tr_gt, dir_path, symbols_to_test
        print('Usage: \npython run_parser.py <parser_pkl> <segmenter_pkl> <symb_clf_pkl> <out_dir_name> <abs_dir_path> <test_expressions>\n')
        sys.exit(0)

    parser_model = sys.argv[1]
    segmenter_model = sys.argv[2]
    symb_clf_model = sys.argv[3]
    out_dname = sys.argv[4]
    abs_dir_path = sys.argv[5]
    symbols_to_test = sys.argv[6]

    parser_test(parser_model, segmenter_model, symb_clf_model, out_dname, abs_dir_path, symbols_to_test)
