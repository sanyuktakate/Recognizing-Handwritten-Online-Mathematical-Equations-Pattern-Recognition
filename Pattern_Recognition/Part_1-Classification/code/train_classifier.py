'''
@author: Sanyukta Kate, Pratik Bongale
'''

import sklearn.neighbors as skl
from sklearn.ensemble import RandomForestClassifier
import pickle
import numpy as np
import inkml_parser
import sys
import preprocessing
import feature_generation
import os

def train_classifier(model, ds_type, gt_fname, abs_dir_path, symbols_to_train):
    '''
    Trains specified model(kdtree or randomforest) using the symbols specified in "symbols_to_train" file
    :param model: string - 'kdtree' or 'randomforest'
    :param ds_type: string - 'v' for valid symbols or 'v_j' for valid+junk symbols
    :param gt_fname: name of file containing records Annotation,label (ground truth of the training set is "symbols_to_train")
    :param abs_dir_path: The absolute path to the directory where the .inkml files are stored(it can be a root directory)
    :param symbols_to_train: .csv file containing .inkml filenames of symbol files on which model needs to be trained
    :return:
    '''

    # read the ground truth file
    with open(gt_fname, 'r') as gt_file:
        label_map = {}
        for line in gt_file:
            label_id, label = line.strip().split(",")
            label_map[label_id] = label

    # read all the files in the current directory and sub directories recursively and built a dictionary
    file_dictionary = dict()

    for (dirpath, dirnames, filenames) in os.walk(abs_dir_path):

        for file in filenames:
            file_dictionary[file] = os.path.join(dirpath, file)

    with open(symbols_to_train, 'r') as symbols_list:

        features = []
        label_ids = []
        labels = []

        for line in symbols_list:
            fname = line.strip()

            # find the file "fname" recursively
            abs_fpath = file_dictionary[fname]

            data = inkml_parser.parse_file(abs_fpath)

            strokes = data["stroke_list"]
            ui = data["label_id"]

            preprocessing.process_sample(strokes)
            feature_vec = feature_generation.gen_feature_vector(strokes)

            features.append(feature_vec)
            label_ids.append(ui)
            labels.append(label_map[ui])

        X = np.array(features)
        classifier_model = None
        if model == 'kdtree':
            # leaf size is the the min number of samples in a given node
            tree = skl.KDTree(X, leaf_size=60)  # Train the classifier, build the tree
            classifier_model = tree

        elif model == 'randomforest':
            # build RandomForest
            rf = RandomForestClassifier(n_estimators=50, max_depth=20, min_samples_split=3)
            rf.fit(X, labels)
            classifier_model = rf

        # store the model parameters in csv file (kdTree_parameters.csv)
        with open(model+'_'+ds_type+'.pickle', 'wb') as model_file:
            pickle.dump(classifier_model, model_file)

if __name__ == '__main__':

    # model, ds_type, iso_gt_fname, symbols_to_train):
    if len(sys.argv) != 6:
        print('usage: python train_classifier.py <model_name> <ds_type> <gt_fname> <abs_dir_path> <symbols_to_train>')
        sys.exit(0)

    model = sys.argv[1]
    ds_type = sys.argv[2]
    gt_fname = sys.argv[3]
    abs_dir_path = sys.argv[4]  # can be a root directory, files will be searched recursively
    symbols_to_train = sys.argv[5]

    train_classifier(model, ds_type, gt_fname, abs_dir_path, symbols_to_train)