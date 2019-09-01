'''
@author: Sanyukta Kate, Pratik Bongale
'''

import pickle
import numpy as np
import sys
import inkml_parser
import os
import glob
import preprocessing
import feature_generation

def rf_test(model_pickle, abs_dir_path, symbols_to_test):
    '''
    Test Random Forest model using the .inkml symbols in file "symbols_to_test"
    :param model_pickle: The pickled Random Forest model
    :param abs_dir_path: absolute path to the a parent directory where all .inkml files(valid, junk, other) can be found
    :param symbols_to_test: a .csv file containing filenames(.inkml) of symbols to test
    :return:
    '''

    rf_pickle = open(model_pickle, 'rb')
    rf = pickle.load(rf_pickle)

    # read all the files in the current directory and sub directories recursively and built a dictionary
    file_dictionary = dict()

    for (dirpath, dirnames, filenames) in os.walk(abs_dir_path):

        for file in filenames:
            file_dictionary[file] = os.path.join(dirpath, file)

    with open(symbols_to_test, 'r') as symbols_list:

        features = []
        label_ids = []

        # read each symbol file, parse, process, generate features
        for line in symbols_list:
            fname = line.strip()

            abs_fpath = file_dictionary[fname]

            data = inkml_parser.parse_file(abs_fpath)

            if data is not None:
                strokes = data["stroke_list"]
                ui = data["label_id"]

                preprocessing.process_sample(strokes)
                feature_vec = feature_generation.gen_feature_vector(strokes)

                features.append(feature_vec)
                label_ids.append(ui)

        test_features = np.array(features)

        class_lst = rf.classes_

        # predict the probability of this sample being in class 1, class 2 ...
        cls_probabilities = rf.predict_proba(test_features)

        # sort probabilities in descending order and get the class indices in cls_list
        indices = np.argsort(-cls_probabilities)

        # retain the top 30 predictions for all samples
        indices = indices[:, :30]  # index in the class list

        n_r, n_c = indices.shape

        inp_fname = os.path.split(symbols_to_test)[-1]
        with open('rf_output_'+inp_fname, 'w') as results_file:

            for r in range(n_r):
                # find the topmost prediction
                results_file.write(label_ids[r] + ',')

                unique_labels = set()
                # write the top ten results of prediction
                for c in range(n_c):
                    idx = indices[r, c]
                    # get the label of sample at this index in training dataset
                    label = class_lst[idx]  # a tuple (label_id, label)

                    # only put the unique labels in results file
                    if label not in unique_labels:
                        unique_labels.add(label)
                        results_file.write(label + (',' if c < n_c - 1 else ''))

                    if len(unique_labels) == 10:
                        break

                results_file.write('\n')

if __name__ == '__main__':

    if len(sys.argv) != 4:
        # model_pickle, tr_gt, dir_path, symbols_to_test
        print('Usage: \npython rf_classifier.py <model_pickle> <abs_dir_path> <symbols_to_test>\n')
        sys.exit(0)

    # Run rf
    model = sys.argv[1]
    abs_dir_path = sys.argv[2]
    symbols_to_test = sys.argv[3]

    rf_test(model, abs_dir_path, symbols_to_test)