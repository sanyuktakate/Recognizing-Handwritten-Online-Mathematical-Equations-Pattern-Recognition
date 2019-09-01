'''
@author: Sanyukta Kate, Pratik Bongale

Paper referred:
Hu, Lei, and Richard Zanibbi.
"HMM-based recognition of online handwritten mathematical symbols using segmental k-means initialization and a modified pen-up/down feature."
Document Analysis and Recognition (ICDAR), 2011 International Conference on. IEEE, 2011.
'''

# import parse_ink_file
# import visualize_ink
# import numpy as np
# import math
from preprocessing import *

def gen_feature_vector(strokes):

    # expecting only 30 points in each sample
    # for each point, we generate cosine of slope, sin of curvature, normalized y co-ordinate

    # strokes = sample["strokes"]
    feature_vec = []

    for stroke in strokes:

        x = stroke["x"]
        y = stroke["y"]

        n = len(x)

        for i in range(n):      # for each point in stroke, compute the features

            if i >= 2 and i < n-2:  # check if its the initial/end point of a stroke

                prev_pt =   { "x": x[i-2], "y": y[i-2] }
                point =     { "x": x[i], "y": y[i] }
                next_pt =   { "x": x[i+2], "y": y[i+2] }

                # check if prev_pt, point and next_pt are duplicates

                cos_vicinity = cos_of_slope(prev_pt, point, next_pt)
                sin_curvature = sin_of_curvature(prev_pt, point, next_pt)
                normalized_y = y[i]

                feature_vec.extend([cos_vicinity, sin_curvature, normalized_y])
            else:
                cos_vicinity = 0.0
                sin_curvature = 0.0
                normalized_y = y[i]

                feature_vec.extend([cos_vicinity, sin_curvature, normalized_y])
                # feature_vec.extend([normalized_y])

    return feature_vec

def cos_of_slope(prev_pt, point, next_pt):
    '''
    cosine of vicinity slope
    :param prev_pt: previous point on trajectory   x(t-2), y(t-2)
    :param point: current point on trajectory      x(t), y(t)
    :param next_pt: next point on trajectory       x(t+2), y(t+2)
    :return:
    '''

    hor_ax_pt = { "x": prev_pt["x"] + 1.0, "y": prev_pt["y"]}   # define a point on horizontal axes of point prev

    # using the law of cosines: cos(θ) = ( (ab)^2 + (bc)^2 - (ac)^2 ) / 2(ab)(bc)

    a = hor_ax_pt
    b = prev_pt
    c = next_pt

    # distance between a,c and b,c and a,c
    ab = math.hypot(b["x"] - a["x"], b["y"] - a["y"])
    bc = math.hypot(c["x"] - b["x"], c["y"] - b["y"])
    ac = math.hypot(c["x"] - a["x"], c["y"] - a["y"])

    cos_theta = (( (ab**2 + bc**2) - ac**2) / (2 * ab * bc) )

    # angle = math.acos(cos_theta)    # in radians

    # return math.cos(angle)

    return cos_theta

def sin_of_curvature(prev_pt, point, next_pt):
    '''
    sin of curvature at current point
    :param prev_pt: previous point on trajectory   x(t-2), y(t-2)
    :param point: current point on trajectory      x(t), y(t)
    :param next_pt: next point on trajectory       x(t+2), y(t+2)
    :return:
    '''

    # using the law of cosines to find θ: cos(θ) = ( (ab)^2 + (bc)^2 - (ac)^2 ) / 2(ab)(bc)

    a = prev_pt
    b = point
    c = next_pt

    # distance between a,c and b,c and a,c
    ab = math.hypot(b["x"] - a["x"], b["y"] - a["y"])
    bc = math.hypot(c["x"] - b["x"], c["y"] - b["y"])
    ac = math.hypot(c["x"] - a["x"], c["y"] - a["y"])

    cos_theta = (((ab ** 2 + bc ** 2) - ac ** 2) / (2 * ab * bc))

    # acos() takes values only between [-1 , 1]
    if cos_theta < -1 or cos_theta > 1:

        if cos_theta < -1:
            cos_theta = -1
        else:
            cos_theta = 1

    angle = math.acos(cos_theta)  # acos requires values between [-1,1]

    return math.sin(angle)


# if __name__ == '__main__':
#
#     # dir = parse_ink_file.get_current_dir() + "/task2-trainSymb2014/mySymbols/"
#     dir = parse_ink_file.get_current_dir() + "/task2-trainSymb2014/specialFiles/"
#
#     fname = 'iso85785.inkml'
#
#     data = parse_ink_file.parse_file(dir + fname)  # get a list of strokes in a symbol
#
#     strokes = data["stroke_list"]
#
#     remove_duplicates(strokes)
#     smoothing(strokes)
#
#     normalize(strokes)
#     resample(strokes)
#
#     #### feature generation ####
#     sample = dict()
#     sample["strokes"] = strokes
#
#     sample["file"] = fname
#
#     features = gen_feature_vector(sample)
#
#     # count of points in a symbol
#     count = 0
#
#     for stroke in strokes:
#         count += len(stroke["x"])
#         print('Count: ' + str(count))
#
#     # original figure
#     visualize_ink.visualize(dir + fname)
#
#     # modified strokes
#     visualize_ink.visualize_strokes_list(strokes)

