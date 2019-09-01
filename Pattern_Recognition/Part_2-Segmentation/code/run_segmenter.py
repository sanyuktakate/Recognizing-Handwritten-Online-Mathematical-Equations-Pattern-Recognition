'''
@author: Sanyukta Kate, Pratik Bongale
'''

import pickle
import gzip
import ntpath
import sys
import preprocessing, feature_generation
from data_preparation import preprocess
from helper import parse_raw_ink_data
import geometric_features
import shape_context_features
from inkml import Inkml, Segment
import os

def seg_test(seg_model_pkl, clf_model_pkl, abs_dir_path, test_data):
    '''
    Run the segmenter model on math expression(s) filenames listed in the file test_data
    :param seg_model_pkl: file name of pickled segmenter model
    :param clf_model_pkl: file name of pickled isolated symbols classifier model
    :param abs_dir_path: absolute path to test dataset directory(can be a parent directory, .inkml files are recursively found from here)
    :param test_data: a file containing filenames(.inkml) of symbols to test
    :return:
    '''

    # load the binary merge/split classifier
    with gzip.open(seg_model_pkl, 'rb') as seg_pickle:
        seg = pickle.load(seg_pickle)

    # load the isolated symbols classifier
    with gzip.open(clf_model_pkl, 'rb') as clf_pickle:
        clf = pickle.load(clf_pickle)

    # read all the files in the current directory and sub directories recursively and built a dictionary
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

                    # perform preprocessing (smoothing, duplicate removal, resampling)
                    preprocess(file_obj)

                    expressions.append(file_obj)

    # extract features for binary classsifier (restrict to only stroke pairs)
    for ink in expressions:         # for each expression

        nStrokes = len(ink.strokes)

        # if we have only one stroke in the expression, send the value to final classifier
        if not nStrokes > 1:
            seg_id = '2'
            strID = list(ink.strokes.keys())     # will only have a single stroke
            label = get_symbol_label(clf, [ink.strokes[strID[0]]])
            segment = Segment(seg_id, label, set(strID))    # create a segment object for this symbol
            write_to_lg(ink, [segment])      # write identified symbols to label graph file
            continue

        # find whether to merge or split successive strokes
        merged_strokes = list()
        for i in range(nStrokes - 1):
            s1 = ink.strkOrder[i]       # stroke id 1
            s2 = ink.strkOrder[i + 1]

            geo_feature_vec = geometric_features.get_pair_features(ink.strokes[s1], ink.strokes[s2])
            shape_feature_vec = shape_context_features.shape_features(ink.strokes[s1], ink.strokes[s2])

            feature_vec = geo_feature_vec + shape_feature_vec

            # predict using binary classifier
            if seg.predict([feature_vec]) == [1]:  # merge
                merged_strokes.append({s1,s2})

        # find connected components
        res_symbols = list()

        # while merge list is not empty
        while merged_strokes:

            # find all the connected consecutive stroke pairs
            n = len(merged_strokes)
            temp = set()
            i = 1

            while i < n and merged_strokes[i-1].intersection(merged_strokes[i]):
                temp.update(merged_strokes[i-1])
                i += 1

            temp.update(merged_strokes[i-1])
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

        # print(ink.fileName, " : ", res_symbols)
        # print('Processing:', ink.fileName)

        # predict class label for each symbol
        seg_id = max([int(a) for a in ink.strkOrder]) + 1
        segments = list()
        for i, symb_strokes in enumerate(res_symbols):
            seg_id += i
            strID = symb_strokes
            label = get_symbol_label(clf, [ink.strokes[s] for s in symb_strokes])

            # create a segment object for this symbol
            segments.append( Segment(seg_id, label, strID) )

        # write identified symbols to label graph file
        write_to_lg(ink, segments)

    print('Label graph(.lg) files have been saved to directory output_lg')

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


def write_to_lg(ink, seg_list):
    '''
    Writes the segments detected to a .lg file in directory "label_graphs"
    :param seg_list: list of Segment objects
    :return: None
    '''

    lines = list()
    lines.append("# IUD, %s\n" % ink.UI)
    lines.append("# Objects(%d):\n" % len(seg_list))

    # find the object id's for each symbol
    object_count = dict()    # stores the count of each object
    for seg in seg_list:
        if seg.label in object_count:
            object_count[seg.label] += 1
        else:
            object_count[seg.label] = 1

    for seg in seg_list:
        # formulate a object_id for each object
        objId = "{}_{}".format(seg.label.strip("\\"), object_count[seg.label])
        object_count[seg.label] -= 1

        row = ['O', objId.strip(), seg.label.strip(), '1.0']
        for s in seg.strId:
            row.append(s)

        pr_row = ""
        for i, ele in enumerate(row):
            pr_row += ele + (", " if i < (len(row)-1) else "\n")

        lines.append(pr_row)

    out_fname = ntpath.basename(ink.fileName).split(".")[0] + ".lg"

    # create a directory to store the output .lg files
    output_dir = 'output_lg'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(os.path.join(output_dir, out_fname), 'w') as out_file:
        out_file.writelines(lines)

if __name__ == '__main__':

    if len(sys.argv) != 5:
        # model_pickle, tr_gt, dir_path, symbols_to_test
        print('Usage: \npython run_segmenter.py <segmentor_pkl> <classifier_pkl> <abs_dir_path> <test_expressions>\n')
        sys.exit(0)

    seg_model = sys.argv[1]
    clf_model = sys.argv[2]
    abs_dir_path = sys.argv[3]
    symbols_to_test = sys.argv[4]

    seg_test(seg_model, clf_model, abs_dir_path, symbols_to_test)
