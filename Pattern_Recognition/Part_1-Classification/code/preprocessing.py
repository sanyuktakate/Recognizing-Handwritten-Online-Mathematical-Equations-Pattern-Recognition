'''
@author: Sanyukta Kate, Pratik Bongale

Paper referred:
Hu, Lei, and Richard Zanibbi.
"HMM-based recognition of online handwritten mathematical symbols using segmental k-means initialization and a modified pen-up/down feature."
Document Analysis and Recognition (ICDAR), 2011 International Conference on. IEEE, 2011.
'''

# import parse_ink_file
# import visualize_ink
import numpy as np
import math
import scipy.interpolate as sp

def process_sample(strokes):
    '''
    perform all preprocessing steps

    Referred paper:
    HMM-based recognition of online handwritten mathematical symbols using segmental k-means initialization and a modified pen-up/down feature.
    In Document Analysis and Recognition (ICDAR), 2011 International Conference on (pp. 457-462). IEEE.
    Authors: Hu, Lei, and Richard Zanibbi

    :param a list of strokes
    :return:
    '''
    # strokes = sample["strokes"]

    remove_duplicates(strokes)
    smoothing(strokes)
    normalize(strokes)
    resample(strokes)

def remove_duplicates(strokes):
    '''
    remove points which have the same x, y co-ordinates
    :param stroke: a dictionary storing {x:[], y:[], label:[]}
    :return: None
    '''

    for stroke in strokes:
        x = stroke['x']
        y = stroke['y']


        n = len(x)      # number of points in the stroke

        x_new = []
        y_new = []

        unique_pts = set()

        for i in range(n):
            pt = (x[i], y[i])
            if pt not in unique_pts:
                unique_pts.add(pt)
                x_new.append(pt[0])
                y_new.append(pt[1])

        stroke['x'] = x_new
        stroke['y'] = y_new

def smoothing(strokes):
    '''
    smooth the stroke by averaging over all points of stroke to reduce noise
    :param stroke: a dictionary storing {x:[], y:[], label:[]}
    :return: None
    '''

    for stroke in strokes:

        x = stroke['x']
        y = stroke['y']
        # labels = stroke['label']

        n = len(x)  # number of points in the stroke

        x_new = []
        y_new = []

        #
        # leaving the first and last point, every points co-ordinates are repplaced by the average of itself, point before it and after it.
        #

        x_new.append(x[0])
        y_new.append(y[0])

        for i in range(1, n-1): # for each point between the first and last point

            # calculate average
            avg_x = sum([x[i-1], x[i], x[i+1]]) / 3
            avg_y = sum([y[i-1], y[i], y[i+1]]) / 3

            x_new.append(avg_x)
            y_new.append(avg_y)

        x_new.append(x[n-1])
        y_new.append(y[n-1])

        stroke['x'] = x_new
        stroke['y'] = y_new

def normalize(strokes):
    '''
    size normalization: scales the symbol such that y values are between [0,1] retaining the aspect ratio
    :param strokes: a list of all strokes
    :return: None
    '''

    max_y = 0
    min_y = math.inf

    max_x = 0
    min_x = math.inf

    for stroke in strokes:
        x = stroke['x']
        y = stroke['y']

        n = len(x)      # number of points in the stroke

        max_y = max( max(y), max_y )
        min_y = min( min(y), min_y )

        max_x = max( max(x), max_x )
        min_x = min( min(x), min_x )

    #
    # y between [0,1] keeping the same aspect ratio
    #

    if max_y == min_y:
        max_y += 100        # can be any positive number
        min_y -= 100

    asp_ratio = 1 / (max_y - min_y)

    for stroke in strokes:
        x = stroke['x']
        y = stroke['y']

        n = len(x)      # number of points in the stroke

        x_new = []
        y_new = []

        for i in range(n):
            # normalized y (traslate to the center and than divide by magnitude of vector)
            y_norm = (y[i] - min_y) / (max_y - min_y)

            # scale x as per the change in y (to retain the aspect ratio)
            x_norm = (x[i] - min_x) * asp_ratio

            x_new.append(x_norm)
            y_new.append(y_norm)

        stroke['x'] = x_new
        stroke['y'] = y_new

def resample(strokes):

    num_strokes = len(strokes)

    total_sample_pts = 30

    per_stroke = int(total_sample_pts // num_strokes)   # number of sample points per stroke

    adjust_pts = 0                      # the number of extra points to be added to strokes

    if total_sample_pts % num_strokes != 0:
        adjust_pts = total_sample_pts % num_strokes

    # alpha = 13              # resampling distance for trace segmentation

    remove_duplicates(strokes)

    for stroke in strokes:

        # distribute extra points to each stroke
        if adjust_pts > 0:
            k = 1
        else:
            k = 0

        # remove_duplicates(strokes)

        # fit a bspline curve to all the points
        bspline_curve_fitting(stroke, per_stroke + k)

        adjust_pts -= 1

        # Other methods
        ### perform trace segmentation (linear interpolation of points in the stroke)
        # trace_segmentation (stroke, alpha)
        ### compute cubic interpolation
        # cubic_interpolation(stroke, per_stroke + k)

def trace_segmentation(stroke, alpha):

    x = stroke["x"]
    y = stroke["y"]

    n = len(x)

    acc_len = [0] * n   # accumulated stroke length

    # compute accumulated length at each point with resampling distance alpha
    for i in range(1,n):
        a = np.array([x[i], y[i]])
        b = np.array([x[i-1], y[i-1]])

        acc_len[i] = acc_len[i-1] + np.linalg.norm(a-b)    # add the l2 distance

    m = int(math.floor(acc_len[n-1]/alpha))  # num of output points

    x_new = [x[0]]
    y_new = [y[0]]

    j = 1
    for p in range(1, m-1):
        while acc_len[j] < p*alpha:
            j += 1

        c = (p*alpha - acc_len[n-2]) / (acc_len[n-1] - acc_len[n-2])

        x_p = x[n-2] + (x[n-1] - x[n-2]) * c
        y_p = y[n-2] + (y[n-1] - x[n-2]) * c

        x_new.append(x_p)
        y_new.append(y_p)

    x_new.append(x[n-1])
    y_new.append(y[n-1])

    stroke["x"] = x_new
    stroke["y"] = y_new

def cubic_interpolation(stroke, num_pts):
    '''
    Compute a cubic interpolation and sample it to obtain equidistant num_pts from this stroke
    :param stroke:
    :param num_pts:
    :return:
    '''

    x = np.array(stroke["x"])
    y = np.array(stroke["y"])

    x_diff = max(x) - min(x)
    y_diff = max(y) - min(y)

    if x_diff > y_diff:
        cubic_func = sp.interp1d(x, y, kind='linear')

        x_new = np.linspace(min(x), max(x), num_pts)
        y_new = cubic_func(x_new)

    else:
        cubic_func = sp.interp1d(y, x, kind='linear')

        y_new = np.linspace(min(y), max(y), num_pts)
        x_new = cubic_func(y_new)

    # x = np.sort(x)
    # y = np.sort(y)

    # x.sort()
    # y.sort()

    stroke["x"] = x_new
    stroke["y"] = y_new

def bspline_curve_fitting(stroke, num_pts):

    x = stroke["x"]
    y = stroke["y"]

    n = len(x)

    # if 3 or less points on this stroke, generate more points
    if n < 4:

        x_new = []
        y_new = []

        n_pts = 4       # number of points on any line segment after interpolation

        if n == 1:
            x1, y1 = x[0], y[0]
            x2, y2 = x[0] + 0.01, y[0] + 0.01

            x_new = np.linspace(x1, x2, num_pts)
            y_new = np.linspace(y1, y2, num_pts)

            # make sure values are between 0.0 and 1.0, because we are dealing with normalized data here
            # x_new.append(x)
            # x_new.append((x + 0.1))  # x can exceed 1.0, y can't
            # x_new.append( abs(x-0.1) )
            # x_new.append(x)
            # x_new.append(x)
            #
            # y_new.append(y)
            # y_new.append(y)
            # y_new.append(y)
            # y_new.append((y + 0.1) if (y + 0.1) < 1.0 else 1.0)
            # y_new.append( abs(y-0.1) )

            # x_new.extend([ x, x + 0.1, x - 0.1, x, x])  # add points around this point
            # y_new.extend([ y, y, y, y + 0.1, y - 0.1])  # add points around this point

        else:

            # this stroke has atleast 2 points
            for i in range(1, n):

                # add 4 points between every consecutive points
                x1, y1 = x[i-1], y[i-1]
                x2, y2 = x[i], y[i]

                x_pts = np.linspace(x1, x2, n_pts)
                y_pts = np.linspace(y1, y2, n_pts)

                # remove the intersection point of segments
                if i < n-1:
                    x_pts = x_pts[:-1]   # take all points except the last one
                    y_pts = y_pts[:-1]   # take all points except the last one
                x_new.extend(x_pts.tolist())
                y_new.extend(y_pts.tolist())

        x, y = x_new, y_new

    control_pts = np.vstack([x, y])
    tck, u = sp.splprep(control_pts, k=3)   # Bspline of degree 3, make sure m > k (i.e minimum 4 points are provided)
    sample_at = np.linspace(u.min(), u.max(), num_pts)
    x_new, y_new = sp.splev(sample_at, tck)

    stroke["x"] = x_new
    stroke["y"] = y_new

if __name__ == '__main__':
    pass
    # # dir = parse_ink_file.get_current_dir() + "/task2-trainSymb2014/mySymbols/"
    # dir = parse_ink_file.get_current_dir() + "/task2-trainSymb2014/specialFiles/"
    #
    # fname = 'iso46660.inkml'
    #
    # data = parse_ink_file.parse_file( dir + fname )  # get a list of strokes in a symbol
    #
    # strokes = data["stroke_list"]
    #
    # remove_duplicates(strokes)
    # smoothing(strokes)
    #
    # normalize(strokes)
    # resample(strokes)
    #
    # # count of points in a symbol
    # count = 0
    # for stroke in strokes:
    #     count += len(stroke["x"])
    # print('Count: ' + str(count))
    #
    # # original figure
    # visualize_ink.visualize(dir + fname)
    #
    # # modified strokes
    # visualize_ink.visualize_strokes_list(strokes)
