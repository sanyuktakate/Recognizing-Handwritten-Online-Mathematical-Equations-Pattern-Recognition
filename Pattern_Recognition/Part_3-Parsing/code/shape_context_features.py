'''
@author: Sanyukta Kate, Pratik Bongale
'''

import geometric_features
import sys
import math
import numpy as np

def shape_features(stroke1, stroke2):

    stroke1_x_list, stroke1_y_list = stroke1[:, 0], stroke1[:, 1]
    stroke2_x_list, stroke2_y_list = stroke2[:, 0], stroke2[:, 1]

    # 1. stroke pair context features
    stroke_pair_features = stroke_pair_shape_context(stroke1_x_list, stroke1_y_list, stroke2_x_list, stroke2_y_list)

    return stroke_pair_features

def stroke_pair_shape_context(stroke1_x_list, stroke1_y_list, stroke2_x_list, stroke2_y_list):

    # first find the radius of the stroke pair which is the maximum distance between the BB center of the current stroke
    # and any of the point on the two strokes

    # find the bb center of the current stroke
    stroke1_bb_centers = geometric_features.bounding_box(stroke1_x_list, stroke1_y_list)

    # this is the radius of the circle, that is the distance from the center of the circle to the fartest points
    # on both the strokes
    Radius = rd_stpair_shape_feature(stroke1_bb_centers, stroke1_x_list, stroke1_y_list, stroke2_x_list, stroke2_y_list)

    # create a list of points which are to be considered for the stroke pair context feature, that is check the pts
    # in both the strokes if they lie within the stroke pair circle.
    pts_in_circle = {}
    temp_list_x = []
    temp_list_y = []

    pts_in_circle_stroke1 = pts_lying_in_circle(Radius,stroke1_bb_centers, stroke1_x_list, stroke1_y_list)
    pts_in_circle_stroke2 = pts_lying_in_circle(Radius, stroke1_bb_centers, stroke2_x_list, stroke2_y_list)

    temp_list_x.extend(pts_in_circle_stroke1["x"]+pts_in_circle_stroke2["x"])
    temp_list_y.extend(pts_in_circle_stroke1["y"] + pts_in_circle_stroke2["y"])

    pts_in_circle["x"] = temp_list_x
    pts_in_circle["y"] = temp_list_y
    nb_pts_in_circle = len(pts_in_circle["x"])

    # As 5 concentric circles are required, we find r1, r2, r3, r4
    # Partition the big circle into equal 5 concentric circles
    inc_r = Radius/5
    r4 = inc_r   # smallest circle's radius
    r3 = r4+inc_r
    r2 = r3+inc_r
    r1 = r2+inc_r # largest circle's radius


    # The largest radius is "Radius" and the lowest one is "r1"
    # size of bins: 5*12
    bins = compute_bins(pts_in_circle, stroke1_bb_centers, Radius, r1, r2, r3, r4)
    # now the circle with this radius is the stroke pair shape context
    # this circle needs to be broken down into 60 bins, from which the normalized count is got.

    # size of normalized bins: 5*12
    normalized_bins = compute_normalized_counts(bins, nb_pts_in_circle)

    # flatten out the normalized bins: 60 bins to create a feature vector
    normalized_bins = np.array(normalized_bins)
    normalized_bins_reshape = np.reshape(normalized_bins,(60))

    flattened_normalized_bins = list(normalized_bins_reshape)

    # the flattened normalized bins is the stroke pair shape context feature
    return flattened_normalized_bins


def compute_normalized_counts(bins, nb_pts_in_circle):

    for i in range(0, len(bins)):
        for j in range(0, len(bins[0])):

            bins[i][j] = bins[i][j]/nb_pts_in_circle

    return bins

def compute_bins(pts_in_circle, bb_center, Radius, r1, r2, r3, r4):

    # find the slope of the line, where, 1 pt is the pt in the circle and the other pt is the bb_center
    # slope is (y2-y1)/(x2-x1)
    bb_x = bb_center[0]
    bb_y = bb_center[1]

    x_list = pts_in_circle["x"]
    y_list = pts_in_circle["y"]

    bins = []

    for i in range(5):
        bins.append([0] * 12)

    # loop through the list of points in the circle
    for i in range(0, len(x_list)):

        # Find the first, temp_angle math.atan2(y,x)) and converting radians into degrees
        temp_angle = math.degrees(math.atan2(y_list[i]-bb_y, x_list[i]-bb_x))

        if temp_angle<0:
            # the temp_angle is negative and lies in the lower half of the circle
            # computing the actual angle by finding the difference and then adding it to 180
            diff = 180-abs(temp_angle)
            final_angle = 180+diff
        else:
            final_angle = temp_angle

        # Now, find the quadrant (which ranges from 1 to 12)
        prev_angle = 0
        next_angle = 30
        quad_nb = 1

        while True:
            if final_angle == 0:
                quadrant=quad_nb
                break
            elif final_angle>prev_angle and final_angle<=next_angle:
                quadrant = quad_nb
                break
            else:
                prev_angle = next_angle
                next_angle = next_angle+30
                quad_nb+=1

        # The quadrant is found. Now, find the circle in which the point lies in
        # to find the circle, we find the distance of that pt from center and decide which circle it lies in
        pt_radius = math.hypot(bb_x - x_list[i], bb_y - y_list[i])
        circle_Nb = 0
        # Now, check if the distance is lies within which radius of the circles
        if pt_radius<=Radius:
            if pt_radius<=r1:
                if pt_radius<=r2:
                    if pt_radius<=r3:
                        if pt_radius<=r4:
                            circle_Nb = 0
                        else: circle_Nb = 1
                    else: circle_Nb = 2
                else: circle_Nb =3
            else: circle_Nb = 4

        # the circle number and the quadrant are found, and hence, we increment the count of points in that bin by 1
        bins[circle_Nb][quadrant-1]= bins[circle_Nb][quadrant-1]+1

    return bins

def pts_lying_in_circle(radius, bb_center, stroke1_x_list, stroke1_y_list):

    stroke1_bb_x = bb_center[0]
    stroke1_bb_y = bb_center[1]

    # pts_in_circle will have a list of points from stroke 1 and stroke 2 which lie within the circle of stroke pairs.
    pts_in_circle = {}
    pts_in_circle["x"] = None
    pts_in_circle["y"] = None
    x_list = list()
    y_list = list()

    for i in range(0, len(stroke1_x_list)):
        # calculate the distance between bbcenter and stroke1 points
        distance = math.hypot(stroke1_bb_x - stroke1_x_list[i], stroke1_bb_y - stroke1_y_list[i])
        if distance<=radius:
            # put the point of stroke 1 in the pts_in_circle
            x_list.append(stroke1_x_list[i])
            y_list.append(stroke1_y_list[i])

    # for i in range(0, len(stroke2_x_list)):
    #     # calculate the distance between bbcenter and the stroke2 points
    #     distance = math.hypot(stroke1_bb_x - stroke2_x_list[i], stroke1_bb_y - stroke2_y_list[i])
    #     if distance <= radius:
    #         # put the point of stroke 1 in the pts_in_circle
    #         x_list.append(stroke2_x_list[i])
    #         y_list.append(stroke2_y_list[i])

    pts_in_circle["x"] = x_list
    pts_in_circle["y"] = y_list

    return pts_in_circle


def rd_stpair_shape_feature(bb_centers, stroke1_x_list, stroke1_y_list, stroke2_x_list, stroke2_y_list):

    # find the distance of each point in the two strokes from the BB center of stroke1
    max_dist = -sys.maxsize
    stroke1_bb_x = bb_centers[0]
    stroke1_bb_y = bb_centers[1]

    for i in range(len(stroke1_x_list)):  # loop through stroke1
        # euclidean distance
        distance = math.hypot(stroke1_bb_x - stroke1_x_list[i], stroke1_bb_y - stroke1_y_list[i])
        if distance > max_dist:
            max_dist = distance

    for i in range(len(stroke2_x_list)):  # loop through stroke2
        # euclidean distance
        distance = math.hypot(stroke1_bb_x - stroke2_x_list[i], stroke1_bb_y - stroke2_y_list[i])
        if distance > max_dist:
            max_dist = distance

    return max_dist

