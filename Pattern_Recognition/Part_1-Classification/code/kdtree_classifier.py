import pickle
import numpy as np
import sys
import inkml_parser
import os
import preprocessing
import feature_generation


def kdtree_test(model_pickle, tr_gt, abs_dir_path, symbols_to_test):
    '''
    Test KDTree model using the inkml symbols in file "symbols_to_test"
    :param model_pickle: The pickled KDTree model
    :param tr_gt: ground truth file of training symbols which were used to train this model
    :param abs_dir_path: absolute path to the a parent directory where all .inkml files(valid, junk, others) can be found
    :param symbols_to_test: a .csv file containing filenames(.inkml) of symbols to test
    :return:
    '''

    kdtree_model = open(model_pickle, 'rb')
    tree = pickle.load(kdtree_model)

    training_labels = []
    with open(tr_gt, 'r') as training_GT:
        for line in training_GT:
            UI, label = line.strip().split(',')
            training_labels.append(label)

    num_neighbours = 30

    # read all the files in the current directory and sub directories recursively and built a dictionary
    file_dictionary = dict()

    for (dirpath, dirnames, filenames) in os.walk(abs_dir_path):

        for file in filenames:
            file_dictionary[file] = os.path.join(dirpath, file)

    with open(symbols_to_test, 'r') as symbols_list:

        features = []
        label_ids = []

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

        # query the tree for k nearest neighbors
        dist, indices = tree.query(test_features, k=num_neighbours)

        n_r, n_c = indices.shape

        inp_fname = os.path.split(symbols_to_test)[-1]
        with open('kdtree_output_'+inp_fname, 'w') as results_file:

            for r in range(n_r):

                # find the label id for this sample
                results_file.write(label_ids[r] + ',')

                unique_labels = set()

                # write the top ten results of prediction
                for c in range(n_c):
                    idx = indices[r, c]
                    # get the label of sample at this index in training dataset
                    label = training_labels[idx]  # a tuple (label_id, label)

                    if label not in unique_labels:
                        unique_labels.add(label)
                        results_file.write(label + (',' if c < n_c - 1 else ''))

                    if len(unique_labels) == 10:
                        break

                results_file.write('\n')


if __name__ == '__main__':

    if len(sys.argv) != 5:
        # model_pickle, tr_gt, dir_path, symbols_to_test
        print('Usage: \npython kdtree_classifier.py <model_pickle> <training_gt> <abs_dir_path> <symbols_to_test>\n')
        sys.exit(0)

    model = sys.argv[1]
    tr_gt = sys.argv[2]
    abs_dir_path = sys.argv[3]
    symbols_to_test = sys.argv[4]

    kdtree_test(model, tr_gt, abs_dir_path, symbols_to_test)


