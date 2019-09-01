'''
@author: Sanyukta Kate, Pratik Bongale
'''

import os
from bs4 import BeautifulSoup
import math
import sys

def split(symbols_dir_abs_path, gt_fname, ds_type):
    '''
    Split the given directory of files into training and testing dataset
    :param symbols_dir_abs_path: absolute path to the directory where .inkml files which we want to split can be found
    :param gt_fname: file name of ground truth file which must be present in symbols_dir_abs_path
    :param ds_type: dataset type, can be valid('v') or junk('j')
    :return:
    '''

    # read gt file and create GT = {ui:label}
    gt_fname = os.path.join(symbols_dir_abs_path, gt_fname)
    with open(gt_fname, 'r') as gt_file:

        gt_dict = {}

        for line in gt_file:
            ui, label = line.strip().split(",")
            gt_dict[ui] = label

    # get all file names
    dir_files = os.listdir(symbols_dir_abs_path)
    class_dict = dict() # {label : [(fname1,ui), (fname2,ui) ...]}

    # parse each inkml file
    for fname in dir_files:
        if fname.endswith('.inkml'):

            abs_filename = os.path.join(symbols_dir_abs_path, fname)

            ui = get_ui(abs_filename)
            label = gt_dict[ui]

            if label not in class_dict:
                class_dict[label] = []

            class_dict[label].append( (fname, ui) )

    classes = class_dict.keys()

    # for cls in classes:
    #     print(cls + ' : ' + str(len(class_dict[cls])))


    # write the training and testing sets to two different .csv files
    training_perc = 70 / 100

    with open(ds_type+'_tr.csv', "w") as train_f, \
            open(ds_type+'_tst.csv', 'w') as test_f, \
            open(ds_type+'_tr_gt.csv', 'w') as tr_gt_f, \
            open(ds_type+'_tst_gt.csv', 'w') as tst_gt_f:

        for cls in classes:

            cls_files = class_dict[cls]

            nb_cls_samples = len(cls_files)  # number of samples in this class

            train_samples = int(math.floor(training_perc * nb_cls_samples))
            # test_samples = nb_cls_samples - train_samples

            training_set = cls_files[:train_samples]
            testing_set = cls_files[train_samples:]

            for sample in training_set:

                fname, ui = sample
                train_f.write(fname + "\n")
                tr_gt_f.write(ui + ','+ cls + "\n")


            for sample in testing_set:
                fname, ui = sample
                test_f.write(fname + "\n")
                tst_gt_f.write(ui + ',' + cls + "\n")

def get_ui(f_name):
    with open(f_name, "r") as inp_file:
        data = inp_file.read()
        soup = BeautifulSoup(data, 'xml')

        annotation = soup.find(type="UI")
        ui = annotation.string
    return ui

if __name__ == '__main__':


    if len(sys.argv) != 4:
        print('Usage: python splitter.py <symbols-dir> <ground-truth-file-name> <ds-type>')
        sys.exit(0)

    dir_path = sys.argv[1]  # please provide absolute path
    gt_fname = sys.argv[2]
    ds_type = sys.argv[3]

    split(dir_path, gt_fname, ds_type)


