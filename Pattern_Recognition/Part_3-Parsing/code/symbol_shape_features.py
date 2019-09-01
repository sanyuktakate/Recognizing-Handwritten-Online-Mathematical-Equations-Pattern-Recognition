'''
@author: Sanyukta Kate, Pratik Bongale
'''

import shape_context_features
import math
import numpy as np

def parzen_features(s1_id, s2_id, seg_map, seg_strokes):
    '''

    :param s1_id: parent symbol
    :param s2_id: child symbol
    :param seg_map:
    :param seg_strokes:
    :return:
    '''

    parzen_shape_features = list()

    # finding the bounding box centers using the symbol ids
    s1_bb_center = seg_map[s1_id]
    s2_bb_center = seg_map[s2_id]

    bb_centers = [(s1_bb_center[0] + s2_bb_center[0])/2, (s1_bb_center[1] + s2_bb_center[1])/2]


    # Radius of the largest/outermost circle using the center
    Radius = shape_context_features.rd_stpair_shape_feature(bb_centers, seg_strokes[s1_id]["x"],seg_strokes[s1_id]["y"], seg_strokes[s2_id]["x"],seg_strokes[s2_id]["y"])

    #  Find the points lying in the parent symbol using radius of the big circle)
    pts_in_circle_parent = shape_context_features.pts_lying_in_circle(Radius, bb_centers, seg_strokes[s1_id]["x"],seg_strokes[s1_id]["y"])

    # find the points lying in the child symbol using Radius
    pts_in_circle_child = shape_context_features.pts_lying_in_circle(Radius, bb_centers, seg_strokes[s2_id]["x"], seg_strokes[s2_id]["y"])

    # As 5 concentric circles are required, we find r1, r2, r3, r4
    # Partition the big circle into equal 5 concentric circles
    inc_r = Radius / 5
    r4 = inc_r  # smallest circle's radius
    r3 = r4 + inc_r
    r2 = r3 + inc_r
    r1 = r2 + inc_r  #  second largest circle's radius

    radius_list = [Radius, r4, r3, r2, r1] # decreasing order
    # total bins is 5*12 = 60 bins
    parent_bins = compute_bins(pts_in_circle_parent, bb_centers, radius_list)

    child_bins = compute_bins(pts_in_circle_child, bb_centers, radius_list)

    parzen_shape_features.extend(parent_bins)
    parzen_shape_features.extend(child_bins)

    return parzen_shape_features

def compute_bins(pts_in_circle, bb_centers, radius_list):

    # 1. compute the center point of every bin
    # 2. As soon as you compute the center point of every bin, calculate the gaussian kernel for the center point using
    # using all the points in the given circle
    # 3. Add this distributed value into the parzen window feature

    # Calculating the center point of the bin
    # 1. first find the peripheral points of the angle

    bins = list()
    # in degrees
    actual_angle = 0
    inc_angle = 15

    x_std = calculate_standard_deviation(pts_in_circle["x"])
    y_std = calculate_standard_deviation(pts_in_circle["y"])

    while actual_angle<360:

        # increament the temp_angle
        temp_angle = actual_angle+inc_angle

        peripheral_points = []  # list of dictionaries which will store all the peripheral points of temp_angle
        # print("bb_centers")
        # print(bb_centers)
        # # find the peripheral points of the 5 circles of the temp_angle. r1 is largest, r4 is smallest
        for r in radius_list:
            # Remeber to change stuff according to the center of bb_centers
            peripheral_points.append(get_peripheral_point(bb_centers, r, temp_angle))  # list of dictionaries

        peripheral_points.append({"x": bb_centers[0], "y":bb_centers[1]})

        # Now find the centers of every bin in that sector using the peripheral points
        for index in range(0, len(peripheral_points)-1):
            center = dict()
            center["x"] = (peripheral_points[index]["x"]+peripheral_points[index+1]["x"])/2
            center["y"] = (peripheral_points[index]["y"]+peripheral_points[index+1]["y"])/2

            # using this center find the gaussian distribution of that bin and append it to the bins list
            bins.append(find_gaussian_distribution(center, pts_in_circle, x_std, y_std))

        actual_angle+=30
    return bins

def find_gaussian_distribution(center, pts_in_circle, x_std, y_std):

    sum = 0

    for pt in range(0, len(pts_in_circle)):   # loop through the pts in circle

        x_part = calculate_sub_expression(center["x"], pts_in_circle["x"][pt], x_std)
        y_part = calculate_sub_expression(center["y"], pts_in_circle["y"][pt], y_std)

        # multiply both the parts
        sum+=(x_part * y_part)

    # take average
    return sum/len(pts_in_circle)

def calculate_standard_deviation(points):

    return np.std(np.asarray(points))

def calculate_sub_expression(center_point, symbol_pt, std):

        # handle the case when all points accumulate at the mean
        if std == 0:
            std = 0.1

        # calculate and return the value above
        exponent = math.exp(-((math.pow((symbol_pt-center_point), 2))/(2*std*std)))

        return ((1/(math.sqrt(2*math.pi) * std))*exponent)

def get_peripheral_point(bb_centers, radius, temp_angle):

    # peri_pts is a dictionary with "x" and "y" coordinates
    peri_pts = dict()

    peri_pts["x"] = bb_centers[0]+(radius* math.cos(math.degrees(temp_angle)))
    peri_pts["y"] = bb_centers[1]+(radius*math.sin(math.degrees(temp_angle)))
    return peri_pts
