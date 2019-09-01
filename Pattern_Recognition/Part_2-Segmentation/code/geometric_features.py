# Name: Sanyukta Sanjay Kate, Pratik Bongale

import math
import matplotlib.pyplot as plt
import sys

# get the geometric features for the stroke pairs
def get_pair_features(stroke1, stroke2):

    stroke1_x_list, stroke1_y_list = stroke1[:, 0], stroke1[:, 1]
    stroke2_x_list, stroke2_y_list = stroke2[:, 0], stroke2[:, 1]

    # find the bounding box centers of the two strokes. A list is returned which has two BB_center_coordinates
    stroke1_bb_centers = bounding_box(stroke1_x_list,stroke1_y_list)
    stroke2_bb_centers = bounding_box(stroke2_x_list,stroke2_y_list)

    stroke1_avg_centers = averaged_centers(stroke1_x_list,stroke1_y_list)
    stroke2_avg_centers = averaged_centers(stroke2_x_list,stroke2_y_list)

    # 1. Horizontal Distance between bounding box centers
    horizontal_distance = horizontal_distance_BB(stroke1_bb_centers, stroke2_bb_centers)
    # 2. Vertical distance between bounding box centers
    vertical_distance = vertical_distance_BB(stroke1_bb_centers, stroke2_bb_centers)
    # 3. The euclidean distane between the bounding box centers
    distance = distance_BB(stroke1_bb_centers, stroke2_bb_centers)
    # 4. distance between averaged centers
    distance_avg_centers = distance_averaged_centers(stroke1_avg_centers, stroke2_avg_centers)
    # 5. writing_slope feature
    slope = writing_slope(stroke1_x_list, stroke1_y_list,stroke2_x_list, stroke2_y_list )

    # 6 Maximal distance of two points, where the two points are on two different strokes
    maximum_distance = maximal_distance(stroke1_x_list, stroke1_y_list,stroke2_x_list, stroke2_y_list)

    # the feature vector for one stroke pair
    features = [horizontal_distance, vertical_distance, distance, distance_avg_centers, slope, maximum_distance]

    return features

def bounding_box(stroke_x_list,stroke_y_list):

    # find min, max of the x_list and the y_list
    min_x = min(stroke_x_list)
    max_x = max(stroke_x_list)

    min_y = min(stroke_y_list)
    max_y = max(stroke_y_list)

    # find width and height of the bounding box of that stroke
    width = max_x-min_x
    height = max_y-min_y

    corner_point = [min_x, min_y]

    # Center of Bounding Box
    bb_center_point = [(corner_point[0]+(width/2)), (corner_point[1]+(height/2))]


    return bb_center_point # returns a list with 2 coordinates

def averaged_centers(stroke_x_list,stroke_y_list):

   sum_x = 0
   sum_y = 0
   for x in stroke_x_list:
      sum_x+=x

   for y in stroke_y_list:
      sum_y+=y

   average_centers = [sum_x/len(stroke_x_list), sum_y/len(stroke_y_list)]


   return average_centers


# Finding the horizontal distance between the Bounding Box centers
def horizontal_distance_BB(stroke1_bb_centers, stroke2_bb_centers):


    x_coord_BB_stroke1 = stroke1_bb_centers[0]
    x_coord_Bb_stroke2 = stroke2_bb_centers[0]

    return abs(x_coord_BB_stroke1-x_coord_Bb_stroke2)

# Finding the verticle distance between the Bounding Box centers
def vertical_distance_BB(stroke1_bb_centers, stroke2_bb_centers):

    y_coord_BB_stroke1 = stroke1_bb_centers[1]
    y_coord_Bb_stroke2 = stroke2_bb_centers[1]

    return abs(y_coord_BB_stroke1 - y_coord_Bb_stroke2)

# Finding the euclidean distance between the bounding box centers
def distance_BB(stroke1_bb_centers, stroke2_bb_centers):

    # calculate the distance between the centers of the Bounding Boxes

    distance = math.hypot(stroke2_bb_centers[0] - stroke1_bb_centers[0] , stroke2_bb_centers[1] - stroke1_bb_centers[1])
    return distance

# Finding the average distance of the centers of the two strokes using the euclidean distance formula
def distance_averaged_centers(stroke1_avg_centers, stroke2_avg_centers):

    distance = math.hypot(stroke2_avg_centers[0] - stroke1_avg_centers[0], stroke2_avg_centers[1] - stroke1_avg_centers[1])
    return distance

def writing_slope(stroke1_x_list, stroke1_y_list,stroke2_x_list, stroke2_y_list):

    # calculate the writing slope which is the angle between the horizontal line & the line connecting the last pt of
    # of current stroke and the first pt of next stroke

    # compute the three points required, that is the last point of current stroke (stroke1),
    # the first point of the next stroke (stroke2), and the point on horizontal line emerging
    # from the last point of stroke1
    stroke1_last_pt = {"x": stroke1_x_list[-1], "y": stroke1_y_list[-1]}
    stroke2_first_pt = {"x": stroke2_x_list[0], "y": stroke2_y_list[0]}
    pt_hor_line = {"x" : stroke1_last_pt["x"]+1, "y": stroke1_last_pt["y"]}


    # here, stroke1_last_pt is 'a', stroke2_first_pt is 'b', pt_hor_line is 'c'
    # so, the distances to be computed are ab, bc, ac
    a = stroke1_last_pt
    b = stroke2_first_pt
    c = pt_hor_line

    #cases where the last point of the current (stroke1) stroke and the first point(stroke2) of the next stroke
    #then, we get ab distance to be zero, because beth the points will be the same, thus leading to divide by zero issue.

    # find the distance of the three points from each other using euclidean distance
    # distance between a,c and b,c and a,c
    ab = math.hypot(b["x"] - a["x"], b["y"] - a["y"])
    bc = math.hypot(c["x"] - b["x"], c["y"] - b["y"])
    ac = math.hypot(c["x"] - a["x"], c["y"] - a["y"])

    if 2*ab*bc == 0:
        if ab==0 and bc==0:
            ab = 0.1
            bc = 0.1
        else:
            if ab == 0:
                ab = 0.1
            if bc == 0:
                bc=0.1


    # now find the cosine of the angle using the three computed points
    cos_theta = (((ab ** 2 + bc ** 2) - ac ** 2) / (2 * ab * bc))

    # the cos_theta is not within the range of [-1 and 1] then force it to be between that range
    if cos_theta >1 or cos_theta<1:
        cos_theta = min(1, max(cos_theta, -1))

    return(math.acos(cos_theta))

def visualize_strokes_list(strokes):

    plt.figure(2)
    for stroke in strokes:  # for each stroke

        x = stroke["x"]
        y = stroke["y"]

        n = len(x)  # number of points

        plt.scatter(x, y, c='k', marker='o')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Visualize Symbol')
    plt.show()


def maximal_distance(stroke1_x_list, stroke1_y_list,stroke2_x_list, stroke2_y_list):

    # A for loop which takes the points from the first stroke and calculates the distance between that point
    # and every point in the 2nd stroke. The max distance is taken into account

    # To find the maximum distance between the two points in two different strokes, consider
    # every stroke1's point distance from every stroke2's point
    max_dist = -sys.maxsize
    for i in range(len(stroke1_x_list)): # stroke1 index
        for j in range(len(stroke2_x_list)):  # stroke2 index
            # euclidean distance
            distance = math.hypot(stroke2_x_list[j] - stroke1_x_list[i],stroke2_y_list[j] - stroke1_y_list[i])
            if distance>max_dist:
                max_dist = distance

    return max_dist