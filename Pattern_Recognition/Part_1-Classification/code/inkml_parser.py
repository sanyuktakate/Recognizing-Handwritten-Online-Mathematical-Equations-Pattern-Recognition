'''
@author: Sanyukta, Pratik

Parse all .inkml files and create a samples.csv file containing samples with their labels
Process: Read file, parse inkml tags, preprocessing, feature extraction

'''

from bs4 import BeautifulSoup
import os
import sys
import numpy as np
import preprocessing
import feature_generation

def parse(dir, out_fname):
    '''
    Parse all the inkml files in "dir",
    and create a csv file(out_fname) containing one symbol per line
    :param dir: directory containing inkml files
    :param out_fname: the .csv file to which program writes all samples' features
    :return:
    '''

    # get parsed samples dictionary: {"label_id": {"strokes": strokes, "label": label} }
    samples = parse_ink_dir(dir)

    with open(out_fname, 'w') as records:

        for label_id in samples:

            sample = samples[label_id]

            # preprocessing:
            # remove duplicates, smoothing, size normalization, resampling
            preprocessing.process_sample(sample)

            # generate features for every point in the sample
            # cosine of vicinity, normalized y-coord, sin of curvature
            features = feature_generation.gen_feature_vector(sample)

            for x in features:
                records.write(str(x) + ',')

            records.write(label_id + ',')
            records.write(sample['label'])
            records.write('\n')

def parse_ink_dir(dir_name):
    '''
    Read all inkml files in directory, parse the inkml tags,
    perform preprocessing, and feature extraction,
    :param dir_name: relative path of the directory(ex. "d1/d2/")
    :return:
    '''

    dir_abs_path = os.path.abspath(dir_name)

    # samples dictionary: {"label_id": {"strokes": strokes, "label": label} }
    samples = dict()

    # get all file names
    dir_files = os.listdir(dir_abs_path)

    # built the label mappings using GT(ground truth) file
    label_mapping = dict()

    for fname in dir_files:
        if fname.endswith('GT.txt'):
            p = os.path.join(dir_abs_path, fname)
            with open(p, 'r') as labels_file:
                for line in labels_file:
                    line = line.strip().split(",")
                    id = line[0]
                    label = line[1]
                    label_mapping[id] = label
            break

    # parse each inkml file
    for fname in dir_files:
        if fname.endswith('.inkml'):
            filename = os.path.join(dir_abs_path, fname)

            data = parse_file(filename)  # data = {"stroke_list": [], "label_id": "+"}

            if data is not None:
                strokes = data["stroke_list"]
                label_id = data["label_id"]
                label = label_mapping[label_id]

                samples[label_id] = {"strokes": strokes, "label": label}

    return samples

def parse_file(f_name):

    contents = dict()
    contents["stroke_list"] = None
    contents["label_id"] = None

    with open(f_name, "r") as inp_file:
        data = inp_file.read()
        soup = BeautifulSoup(data, 'xml')

        annotation = soup.find(type="UI")
        ui = annotation.string

        traces = soup.find_all('trace')

        # ignored_traces = 0
        ignored_symbol = 0

        trace_list = []
        for i in range(len(traces)):
            trace = traces[i].get_text().strip().split(",")

            x_list = []
            y_list = []

            # handle smaller traces (with 3 or less points)
            if len(trace) < 4:
                x_list, y_list = generate_more_points(trace)

            else:

                for point in trace:

                    point = point.strip().split(" ")

                    # check what information is provided for each point in stroke
                    if len(point) == 3:     # time information
                        x, y, time = [float(p) for p in point]
                    elif len(point) == 2:
                        x, y = [float(p) for p in point]
                    else:
                        sys.exit('More than 3 elements describing a point')

                    x_list.append(x)
                    y_list.append(y)

            trace_dict = dict()
            trace_dict["x"] = x_list
            trace_dict["y"] = y_list

            trace_list.append(trace_dict)

        if len(trace_list) == 0:
            ignored_symbol += 1
            print('Ignored Symbol: ', ui)
            return None

        contents["stroke_list"] = trace_list    # list of trace dictionaries
        contents["label_id"] = ui               # label of this symbol

    return contents

def generate_more_points(trace):
    '''
    If the trace has 1 point generate 4 points around it.
    else if more than 1 points are present in the trace,
    generate more points by interpolation between consecutive points
    :param trace: a list of (x y) points
    :return:
    '''

    # number of points to be generated
    n_pts = 4       # includes the first and last point

    x_list = []
    y_list = []

    if len(trace) == 1:
        x, y = get_point_cord(trace[0])

        x_list.extend([ x, x+1, x-1, x, x])    # add points around this point
        y_list.extend([ y, y, y, y+1, y-1])    # add points around this point
    else:
        # this trace has atleast 2 points
        for i in range(1, len(trace)):

            # add 4 points between every consecutive points
            x1, y1 = get_point_cord(trace[i-1])
            x2, y2 = get_point_cord(trace[i])

            x_pts = np.linspace(x1, x2, n_pts)
            y_pts = np.linspace(y1, y2, n_pts)

            x_list.extend(x_pts.tolist())
            y_list.extend(y_pts.tolist())

    return x_list, y_list

def get_point_cord(point):

    point = point.strip().split(" ")

    # check what information is provided for each point in stroke
    if len(point) == 3:  # time information
        x, y, time = [float(p) for p in point]
    elif len(point) == 2:
        x, y = [float(p) for p in point]
    else:
        sys.exit('More than 3 elements describing a point')

    return x, y

