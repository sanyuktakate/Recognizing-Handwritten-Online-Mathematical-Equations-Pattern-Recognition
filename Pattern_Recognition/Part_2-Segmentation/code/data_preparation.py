import numpy as np
from math import *
import scipy.interpolate as sp

def preprocess(ink):

    # data cleaning
    remove_duplicates(ink)
    smoothing(ink)

    # data preparation
    normalize(ink)
    resample(ink)

    return ink

def remove_duplicates(ink):
    '''
    remove points which have the same x, y co-ordinates
    :param stroke: a dictionary storing {x:[], y:[], label:[]}
    :return: res_pts: the resulting stoke points without any duplicate points
    '''

    for stroke in ink.strokes:
        stroke_pts = ink.strokes[stroke]

        unique_pts = set()
        res_pts = list()

        for pt in stroke_pts:
            if tuple(pt) not in unique_pts:
                unique_pts.add(tuple(pt))
                res_pts.append(pt)

        ink.strokes[stroke] = np.array(res_pts)


def smoothing(ink):
    '''
    smooth the stroke by averaging over all points of stroke to reduce noise
    :param ink: a dictionary storing {x:[], y:[], label:[]}
    :return: res_pts: the resulting smooth stoke points
    '''

    for stroke in ink.strokes:
        stroke_pts = ink.strokes[stroke]
        nPoints = stroke_pts.shape[0]
        res_pts = list()

        # leaving the first and last point, every points co-ordinates are repplaced by the average of itself, point before it and after it.
        res_pts.append(stroke_pts[0])
        for i in range(1, nPoints - 1):
            avg = (stroke_pts[i - 1] + stroke_pts[i] + stroke_pts[i + 1]) / 3
            res_pts.append(avg)

        res_pts.append(stroke_pts[-1])

        ink.strokes[stroke] = np.array(res_pts)

def normalize(ink):
    '''
    size normalization: scales the symbol such that y values are between [0,100] retaining the aspect ratio
    :param strokes: a list of all strokes
    :return: res_pts: the resulting normalized stoke points
    '''

    # find the range of x and y co-ordinates
    min_x, max_x = inf, 0
    min_y, max_y = inf, 0

    for s in ink.strokes:

        pts = ink.strokes[s]

        min_y = min( pts[:, 1].min(), min_y )
        max_y = max( pts[:, 1].max(), max_y )

        min_x = min( pts[:, 0].min(), min_x )
        max_x = max( pts[:, 0].max(), max_x )

    # scale y to range [0,100] retaining aspect ratio
    if max_y == min_y:
        max_y += 100  # can be any positive number
        min_y -= 100

    asp_ratio = 1 / (max_y - min_y)

    for s in ink.strokes:

        pts = ink.strokes[s]

        # normalized y (traslate to the center and than divide by magnitude of vector)
        pts[:, 1] = ((pts[:, 1] - min_y) / (max_y - min_y)) * 100

        # scale x as per the change in y (to retain the aspect ratio)
        pts[:, 0] = ((pts[:, 0] - min_x) * asp_ratio) * 100

        ink.strokes[s] = pts

def resample(ink):

    # points per stroke
    nPoints = 30

    # bspline curve requires that consecutive points are not identical
    remove_duplicates(ink)

    for s in ink.strokes:

        stroke_pts = ink.strokes[s]

        # fit a bspline curve over all points
        new_pts = bspline_curve_fitting(stroke_pts, nPoints)

        ink.strokes[s] = new_pts

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

# https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.splprep.html
# https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.splev.html
def bspline_curve_fitting(stroke, num_pts):

    x = stroke[:, 0]
    y = stroke[:, 1]

    n = len(x)
    bspline_degree = 3  # degree of the spline

    # # indices where consec elements are not identical
    # if np.where(np.abs(np.diff(x)) + np.abs(np.diff(y)) <= 0)[0].size != 0:
    #     print('Duplicate consecutive points identified')

    # if 3 or less points on this stroke, generate more points
    if n < bspline_degree + 1:

        x_new = []
        y_new = []

        n_pts = 4       # number of points on any line segment after interpolation

        if n == 1:
            x1, y1 = x[0], y[0]
            x2, y2 = x[0] + 0.01, y[0] + 0.01

            x_new = np.linspace(x1, x2, num_pts)
            y_new = np.linspace(y1, y2, num_pts)

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

    # bspline errors out if two consecutive elements are identical
    # https://stackoverflow.com/questions/47948453/scipy-interpolate-splprep-error-invalid-inputs
    # non_ident_ind = np.where(np.abs(np.diff(x)) + np.abs(np.diff(y)) > 0) # indices where consec elements are not identical
    # x = np.r_[x[non_ident_ind], x[-1]] # concatenate elements with the last element
    # y = np.r_[y[non_ident_ind], y[-1]]

    control_pts = np.vstack([x, y])
    tck, u = sp.splprep(control_pts, k=bspline_degree)   # Bspline of degree 3, make sure m > k (i.e minimum 4 points are provided)
    sample_at = np.linspace(u.min(), u.max(), num_pts)
    x_new, y_new = sp.splev(sample_at, tck)

    new_stroke = np.stack([x_new, y_new], axis=1)

    return new_stroke

if __name__ == '__main__':
    
    
    from inkml import Inkml
    from helper import parse_raw_ink_data
    import matplotlib.pyplot as plt

    ink = Inkml('dataset/TrainINKML/HAMEX/formulaire024-equation050.inkml')

    parse_raw_ink_data(ink)
    preprocess(ink)

    # show how the expression looks like
    for s in ink.strokes:
        stroke = ink.strokes[s]
        plt.scatter(stroke[:, 0], stroke[:, 1], c='k', marker='+')

    # # perform resampling and curve fitting
    # resample(ink)
    #
    # # see the fit expression
    # for s in ink.strokes:
    #     stroke = ink.strokes[s]
    #     plt.scatter(stroke[:, 0], stroke[:, 1], c='r', marker='*')

    plt.gca().invert_yaxis()
    plt.show()