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
from helper import parse_raw_ink_data, read_lg
from data_preparation import preprocess
from symbol_features import *
from inkml import Inkml

def train_parser(model, out_fname, dir_path_inkml, dir_path_lg, training_fname):
    '''
    Trains specified model with the math expressions specified in "training_fname" file
    :param model: str - 'kdtree' or 'rf' or 'svm'
    :param out_fname: str - name for output model pickle file: model_<out_fname>.pklz
    :param dir_path_inkml: str - directory path of .inkml files needed for training
    :param dir_path_lg: str - directory path of .lg ground truth files
    :param training_fname: str - a file listing all inkml filenames on which parser will be trained
    :return:
    '''

    # read all inkml files in the current directory and sub directories recursively(including softlinks) and build a dictionary
    file_path_ink = dict()
    for (dirpath, dirnames, filenames) in os.walk(dir_path_inkml, followlinks=True):

        for file in filenames:
            file_path_ink[file] = os.path.join(dirpath, file)

    # read all lg files
    file_path_lg = dict()
    for (dirpath, dirnames, filenames) in os.walk(dir_path_lg, followlinks=True):

        for file in filenames:
            file_path_lg[file] = os.path.join(dirpath, file)

    # convert each file to an inkml object
    with open(training_fname, 'r') as expr_files:

        expressions = dict()  # { inkml_object: symb_pair_dictionary }

        for fname in expr_files:
            fname = fname.strip()

            # only consider .inkml files
            if fname.endswith('.inkml'):

                file = fname.split(".")[0]

                abs_fpath_ink = file_path_ink[file + ".inkml"]
                abs_fpath_lg = file_path_lg[file + ".lg"]

                try:
                    # create Inkml object
                    ink_obj = Inkml(abs_fpath_ink)
                    symb_pairs_gt = read_lg(abs_fpath_lg, ink_obj)  # Example: { (s1,s2) : 'Right', (s3,s4) : 'Sup'}

                except:
                    print('error occured in file: ' + fname)
                    continue

                # process stroke points str -> 2d np-array
                parse_raw_ink_data(ink_obj)

                # perform preprocessing (smoothing, duplicate removal, resampling)
                preprocess(ink_obj)

                expressions[ink_obj] = symb_pairs_gt

        print('Pre-processing complete.')

    features = []
    labels = []

    # from tqdm import tqdm
    for ink in expressions:
        symb_pairs_gt = expressions[ink]       # get all symbol relations in ground truth

        for s_pair in symb_pairs_gt:

            l = symb_pairs_gt[s_pair]
            s1, s2 = s_pair
            f_vector = get_symb_features(s1, s2, ink)     # 66 features
            features.append(f_vector)
            labels.append(l)

    X = np.vstack(features)
    y = np.array(labels)

    print('Feature Extraction Complete.')

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
    output_pickle_fname = model + '_' + out_fname + '.pklz'
    with gzip.open(output_pickle_fname, 'wb') as model_file:
        pickle.dump(classifier_model, model_file)

    print('Trained', model, 'model stored in file: ', output_pickle_fname)

if __name__ == '__main__':

    if len(sys.argv) != 6:
        print('usage: python train_segmentor.py <model_name> <output_keyword> <inkml_dir_path> <lg_dir_path> <expr_to_train>')
        sys.exit(0)

    model_name = sys.argv[1]
    output_keyword = sys.argv[2]
    inkml_dir_path = sys.argv[3]    # can be a root directory, files will be searched recursively
    lg_dir_path = sys.argv[4]
    expr_to_train = sys.argv[5]

    train_parser(model_name, output_keyword, inkml_dir_path, lg_dir_path, expr_to_train)

    # nohup python train_classifier.py rf all p2_files/symb_gt.txt /home/pratik/Desktop/PR/Projects/p1/dataset/ p2_files/symb_file_names.txt &