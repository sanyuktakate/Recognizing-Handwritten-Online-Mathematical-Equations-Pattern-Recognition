'''
@author: Sanyukta Kate, Pratik Bongale
'''

import sklearn.neighbors as skl
from sklearn.ensemble import RandomForestClassifier
import pickle
import gzip
import numpy as np
import sys
import os
from helper import parse_raw_ink_data
from data_preparation import preprocess
import geometric_features, shape_context_features
from inkml import Inkml
from los_graph import Graph

def train_segmenter(model, ds_type, abs_dir_path, expr_fnames):
    '''
    Trains specified model with the math expressions specified in "expressions_list" file
    :param model: string - 'kdtree' or 'randomforest'
    :param ds_type: string - a keyword to uniquely identify output model file
    :param abs_dir_path: The absolute path to the directory where the .inkml files are stored(it can be a root directory)
    :param expr_fnames: file containing .inkml filenames on which model needs to be trained
    :return:
    '''

    # read all the files in the current directory and sub directories recursively(including softlinks) and build a dictionary
    file_dictionary = dict()
    for (dirpath, dirnames, filenames) in os.walk(abs_dir_path, followlinks=True):

        for file in filenames:
            file_dictionary[file] = os.path.join(dirpath, file)

    # convert each file to an inkml object
    with open(expr_fnames, 'r') as expr_files:

        expressions = list()    # list of inkml objects

        for fname in expr_files:
            fname = fname.strip()

            # only consider .inkml files
            if fname.endswith('.inkml'):

                abs_file_path = file_dictionary[fname]

                try:
                    # create Inkml object
                    file_obj = Inkml(abs_file_path)
                except:
                    print('error occured in file: ' + fname)
                    continue

                # print('File being processed: ', file_obj.fileName)

                # process stroke points str -> 2d np-array
                parse_raw_ink_data(file_obj)

                # perform preprocessing (smoothing, duplicate removal, resampling)
                preprocess(file_obj)

                expressions.append(file_obj)

        print('Pre-processing complete.')

    # extract features for binary classsifier (merge/split)
    X = []
    y = []
    for ink in expressions:     # for each expression

        nStrokes = len(ink.strokes)
        features = list()       # features of all stroke pairs in ink
        labels = list()         # associated labels (merge/split)

        # don't do anything if we have an expression with only one stroke
        if not nStrokes > 1:
            continue

        for i in range(nStrokes-1):
            s1 = ink.strkOrder[i]
            s2 = ink.strkOrder[i+1]

            geo_feature_vec = geometric_features.get_pair_features(ink.strokes[s1], ink.strokes[s2])
            shape_feature_vec = shape_context_features.shape_features(ink.strokes[s1], ink.strokes[s2])

            features.append(geo_feature_vec + shape_feature_vec)

            # if this pair is a part of any segment, then gt will be 1-Merge, else, 0-Split
            merge = False
            for s in ink.segments:  # for each symbol in expression
                symb_strokes = ink.segments[s].strId
                if len({s1,s2} - symb_strokes) == 0:    # empty set
                    merge = True
                    break

            labels.append(int(merge))

        # # create a line of sight graph for expression
        # los_graph = Graph(ink)
        #
        # for n1 in los_graph.nodes:
        #     s1 = los_graph.nodes[n1]        # get node for s1
        #
        #     for nbor in los_graph.edges[n1]:
        #         s2 = los_graph.nodes[nbor]  # get node for s2
        #
        #         feature_vec = stroke_pairs.get_pair_features(s1.stroke_pts, s2.stroke_pts)
        #         features.append(feature_vec)
        #
        #         # if this pair is a part of any segment, then gt will be 1-Merge, else, 0-Split
        #         merge = False
        #         for s in ink.segments:  # for each symbol in expression
        #             symb_strokes = ink.segments[s].strId
        #             if len({s1.stroke_id, s2.stroke_id} - symb_strokes) == 0:  # empty set
        #                 merge = True
        #                 break
        #
        #         labels.append(int(merge))

        X.extend(features)
        y.extend(labels)

    X = np.vstack(X)
    y = np.array(y)

    classifier_model = None

    print('Training', model, 'classifier')
    if model == 'kdtree':
        # leaf size is the the min number of samples in a given node
        tree = skl.KDTree(X, leaf_size=60)  # Train the classifier, build the tree
        classifier_model = tree

    elif model == 'rf':
        rf = RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_split=10)
        rf.fit(X, y)
        classifier_model = rf

    elif model == 'svm':
        pass

    print('Training complete')

    # store the model parameters in csv file (kdTree_parameters.csv)
    output_pickle_fname = model + '_' + ds_type + '.pklz'
    with gzip.open(output_pickle_fname, 'wb') as model_file:
        pickle.dump(classifier_model, model_file)

    print('Trained', model, 'model stored in file: ', output_pickle_fname)

if __name__ == '__main__':

    # model, ds_type, iso_gt_fname, symbols_to_train):
    if len(sys.argv) != 5:
        print('usage: python train_segmentor.py <model_name> <output_keyword> <inkml_dir_path> <expr_to_train>')
        sys.exit(0)

    model_name = sys.argv[1]
    output_keyword = sys.argv[2]
    inkml_dir_path = sys.argv[3]    # can be a root directory, files will be searched recursively
    expr_to_train = sys.argv[4]

    train_segmenter(model_name, output_keyword, inkml_dir_path, expr_to_train)

    # nohup python train_classifier.py randomforest all p2_files/symb_gt.txt /home/pratik/Desktop/PR/Projects/p1/dataset/ p2_files/symb_file_names.txt &