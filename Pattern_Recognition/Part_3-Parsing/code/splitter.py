'''
@author: Sanyukta Kate, Pratik Bongale
'''

import os
import sys
import math
import random
import ntpath
from inkml import Inkml

def split(ink_files_dir):
    '''
    Split the given directory of files into training and testing dataset
    :param symbols_dir_abs_path: absolute path to the directory where .inkml files which we want to split can be found
    :return: None
    '''

    # read all the files in the current directory and sub directories recursively and built a dictionary
    expressions = list()

    for (dirpath, dirnames, filenames) in os.walk(ink_files_dir, followlinks=True):

        for fname in filenames:

            if fname.endswith('.inkml'):

                file_path = os.path.join(dirpath, fname)

                try:
                    file_obj = Inkml(file_path)
                except:
                    print('error occured in file: ' + fname)
                    continue

                expressions.append(file_obj)

    symb_distribution, _ = get_symbols_dist(expressions)

    train_set, test_set = get_best_approximation(symb_distribution, expressions)

    save_data(train_set, test_set)

def get_symbols_dist(ink_obj_lst):
    '''
    Compute the frequency and probability distribution of symbols in data given as input
    :param ink_obj_lst: A list of Inkml objects
    :return: symbol frequencies and probability distribution of symbols
    '''

    res = dict()        # result probabilities of each symbol
    symb_count = dict() # keeps count of symbols
    total_count = 0

    # count the frequency of each symbol in the dataset of expressions
    for expr in ink_obj_lst:   # for each expression

        segment_lst = expr.segments

        for segment in segment_lst:     # for each segment in expression
            symbol = segment_lst[segment].label
            if symbol in symb_count:
                symb_count[symbol] += 1
            else:
                symb_count[symbol] = 1

            total_count += 1

    # compute symbol probability distribution
    for symbol in symb_count:
        res[symbol] = symb_count[symbol] / total_count
        # res.append(symb_count[symbol] / total_count)

    return symb_count, res

def get_best_approximation(all_symb_count, expressions):
    '''
    Find the best approximation of symbol distribution given as input using KL-divergence
    :param all_symb_count: a dictionary string the count of each symbol
    :param expressions: a list of Inkml objects
    :return: training_set, test_set: two lists with respective inkml objects
    '''
    train_symb_ct = dict()
    total_tr = 0

    test_symb_ct = dict()
    total_tst = 0

    # split all symbols into two parts 2/3 and 1/3
    for symbol in all_symb_count:
        nb_tr_symbols = math.ceil(all_symb_count[symbol] * 0.67)
        train_symb_ct[symbol] = nb_tr_symbols
        total_tr += nb_tr_symbols

        nb_tst_symbols = all_symb_count[symbol] - nb_tr_symbols
        test_symb_ct[symbol] = nb_tst_symbols
        total_tst += nb_tst_symbols

    # compute two distributions P_training, P_testing
    pd_trainset = dict()  # expected probability distribution of training set over all symbols
    pd_testset = dict()  # expected probability distribution of test set over all symbols

    for symbol in all_symb_count.keys():

        pd_trainset[symbol] = train_symb_ct[symbol] / total_tr
        pd_testset[symbol] = test_symb_ct[symbol] / total_tst

        if pd_trainset[symbol] == 0.0:
            pd_trainset[symbol] += 0.000001
        elif pd_testset[symbol] == 0.0:
            pd_testset[symbol] += 0.000001

    # shuffle expressions in the inkml expressions list and do a grid search for best approximation of ideal probability distribution
    iterations = 1000
    nb_expressions = len(expressions)
    nb_tr_expressions = math.ceil(nb_expressions * (2 / 3))
    nb_tst_expressions = nb_expressions - nb_tr_expressions

    # best_kld = 0.0003229711380424644        # best divergence obtained for given data
    # best_seed = 176                         # randomizer seed to get this distribution

    best_kld = math.inf
    best_seed = 0

    # find the best split
    for seed in range(iterations):

        expressions_copy = expressions.copy()

        random.Random(seed).shuffle(expressions_copy)

        # split shuffled expressions into 2/3rd and 1/3rd
        training_data = expressions_copy[ 0 : nb_tr_expressions ]
        test_data = expressions_copy[ nb_tr_expressions : ]

        _, pd_approx_trainset = get_symbols_dist(training_data)
        _, pd_approx_testset = get_symbols_dist(test_data)

        missing_symbols = pd_trainset.keys() - pd_approx_trainset.keys()

        for s in missing_symbols:
            pd_approx_trainset[s] = 0.000001

        kld = calc_kldivergence(pd_trainset, pd_approx_trainset)    # calculate information loss (divergence)

        if kld < best_kld:
            best_kld = kld
            best_seed = seed

    print('Best KLD: ' + str(best_kld))
    print('Best Seed: ' + str(best_seed))

    random.Random(best_seed).shuffle(expressions)

    # split best approximation into 2/3rd and 1/3rd
    training_data = expressions[0:nb_tr_expressions]
    test_data = expressions[nb_tr_expressions:]

    # _, pd_approx_trainset = get_symbols_dist(training_data)
    # _, pd_approx_testset = get_symbols_dist(test_data)

    return training_data, test_data

# https://datascience.stackexchange.com/questions/9262/calculating-kl-divergence-in-python
def calc_kldivergence(p, q):

    # p = np.asarray(p)
    # q = np.asarray(q)
    #
    # sum = np.sum(np.where(p != 0, p * np.log(p / q), 0))

    sum = 0
    for i in p:
        sum += p[i] * math.log(p[i]/q[i], 2)

    return sum

def save_data(train_set, test_set):
    '''
    Saves the training and testing data(as filenames) to two files
    :param train_set: The list of expression objects in the training set
    :param test_set: The list of expression objects in the test set
    :return:
    '''

    with open('training_files.txt', 'w') as training_f:
        for ink in train_set:
            training_f.write(ntpath.basename(ink.fileName + "\n" ))

    with open('test_files.txt', 'w') as training_f:
        for ink in test_set:
            training_f.write(ntpath.basename(ink.fileName + "\n" ))

if __name__ == '__main__':

    if len(sys.argv) != 2:
        print('Usage: python splitter.py <symbols-dir>')
        sys.exit(0)

    dir_path = sys.argv[1]  # please provide absolute path

    split(dir_path)



