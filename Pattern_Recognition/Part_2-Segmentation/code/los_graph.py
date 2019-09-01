from math import *
from scipy.spatial import ConvexHull
import numpy as np
import math

class Node(object):

    __slots__ = ('stroke_id', 'stroke_pts', 'bb_center')

    def __init__(self, *args):
        if len(args) == 2:
            self.stroke_id = args[0]
            self.stroke_pts = args[1]
            self.bb_center = self.get_bb_center()

    def get_bb_center(self):

        # find min, max of the x co-ordinates and the y co-ordinates

        stroke_x = self.stroke_pts[:, 0]
        stroke_y = self.stroke_pts[:, 1]

        min_x = stroke_x.min()
        max_x = stroke_x.max()

        min_y = stroke_y.min()
        max_y = stroke_y.max()

        # find width and height of the bounding box of that stroke
        width = max_x - min_x
        height = max_y - min_y

        corner_point = [min_x, min_y]

        # Center of Bounding Box
        bb_center_point = (corner_point[0] + (width / 2)), (corner_point[1] + (height / 2))

        # stroke["bb_center"] = bb_center_point
        return tuple(bb_center_point)  # returns a list with 2 coordinates

class Graph(object):
    """Class to reprsent a stroke graph"""

    __slots__ = ('nodes', 'edges', 'ink_obj')

    def __init__(self, *args):
        if len(args) == 1:
            self.ink_obj = args[0]
            self.nodes = dict()     # {'1': array([[x1,y1], [x2,y2] ...])}

            for s in self.ink_obj.strkOrder:
                self.nodes[s] = Node(s, self.ink_obj.strokes[s])

            self.edges = dict()     # {'1': ['2', '3', '7'], '2':['1']}

            self.create_graph()


    def create_graph(self):
        '''
        constructing a Line-Of-Sight graph
        References:
        https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7814060 (Line-of-Sight Stroke Graphs and Parzen Shape Context Features for Handwritten Math Formula Representation and Symbol Segmentation)
        http://people.inf.elte.hu/fekete/algoritmusok_msc/terinfo_geom/konyvek/Computational%20Geometry%20-%20Algorithms%20and%20Applications,%203rd%20Ed.pdf

        :return:
        '''

        S = self.ink_obj.strkOrder    # a set of handwritten strokes for an expression

        for s in S:
            s1 = self.nodes[s]      # get a points np array

            sc = s1.bb_center       # get the bounding box center
            U = { (0, 2*pi) }       # all angles are unblocked initially(0-360)

            # sort strokes by increasing distance from s1
            unsorted_s = set(S) - set(s)
            dist_sorted_s = sorted(unsorted_s, key=lambda x1: hypot(sc[0] - self.nodes[x1].bb_center[0], sc[1] - self.nodes[x1].bb_center[1]))

            # determine angles blocked by remaining strokes
            for t in dist_sorted_s:     # for every other stroke in the set
                s2 = self.nodes[t]

                theta_min = math.inf
                theta_max = -math.inf

                for n in self.get_convex_hull(s2.stroke_pts):
                    x_h, y_h = n    # candidate stroke being considered for visibility
                    x_0, y_0 = sc   # bb center

                    plt.plot([x_0, x_h], [y_0, y_h], 'yo-')

                    w = (x_h-x_0, y_h-y_0)
                    h = (1, 0)

                    # if w[0] < 0:
                    #     print('problem: ', w)

                    mag_w = math.sqrt(w[0]**2 + w[1]**2)
                    mag_h = math.sqrt(h[0]**2 + h[1]**2)

                    # find the anngle between vector w and a horizontal vector (1,0)
                    if y_h >= y_0:
                        theta = math.acos( w[0] / mag_w * mag_h )     # acos should always be between -1, 1
                    else:
                        theta = 2*math.pi - math.acos(w[0] / mag_w * mag_h)

                    theta_min = min(theta_min, theta)
                    theta_max = max(theta_max, theta)

                h = (theta_min, theta_max)  # hull interval

                # check remaining unblocked angles and block interval h if available
                V = set()

                for u in U:     # for each unblocked angle interval

                    V.add( self.del_interval(u, h) )    # Union (u-h) for each u in set U

                # if the union of (u-h) is not empty, then add edge
                if len(V) != 0:

                    # add an edge between s1(Node) and s2(Node)
                    if s1.stroke_id in self.edges:
                        self.edges[s1.stroke_id].add(s2.stroke_id)
                    else:
                        self.edges[s1.stroke_id] = set(s2.stroke_id)

                    # add an edge between s2(Node) and s1(Node)
                    if s2.stroke_id in self.edges:
                        self.edges[s2.stroke_id].add(s1.stroke_id)
                    else:
                        self.edges[s2.stroke_id] = set(s1.stroke_id)

                # # block the angles covered by stroke s2
                # new_U = set()
                # for u in U:
                #
                #
                #
                #      U = new_U
                #
                #
                #
                #
                # new_U = set()
                # for u in U:
                #
                #     temp_u = u
                #     new_V = set(V)
                #
                #     for v in new_V:
                #
                #         # remove/block the interval v
                #         u_st, u_end = temp_u
                #         h_st, h_end = v
                #
                #         # if h is between u
                #         if u_st <= h_st and h_end <= u_end:
                #
                #             if u_st != h_st:
                #                 new_V.add((u_st, h_st))
                #             if u_end != h_end:
                #                 new_V.add((h_end, u_end))
                #
                #         # if h is outside u
                #         elif u_st > h_end or u_end < h_st:
                #             new_V.add(u)
                #
                #         # if h is partially inside u
                #         elif h_st > u_st:
                #             new_V.add((u_st, h_st))
                #
                #         # if h is partially inside u
                #         elif h_end < u_end:
                #             new_V.add((h_end, u_end))
                #
                #         # if h is exactly equal to u, or h encapsulates u
                #         else:
                #             pass
                #
                #     new_U.update( new_V )
                #
                # U = new_U

                U = V


    def del_interval(self, u, h):
        '''
        Delete interval (u-h)
        :param u: interval 1
        :param h: interval 2
        :return: remainder of removal
        '''

        remainder = set()

        # remove/block the interval h from interval u
        u_st, u_end = u
        h_st, h_end = h

        # if h is between u
        if u_st <= h_st and h_end <= u_end:
            if u_st != h_st:
                remainder.add((u_st, h_st))
            if u_end != h_end:
                remainder.add((h_end, u_end))

        # if h is outside u, add back the unblocked interval as it is
        elif u_st > h_end or u_end < h_st:
            remainder.add(u)

        # if h is partially inside u
        elif h_st > u_st:
            remainder.add((u_st, h_st))

        # if h is partially inside u
        elif h_end < u_end:
            remainder.add((h_end, u_end))

        # if h is exactly equal to u, or h encapsulates u
        else:
            pass

        return remainder


    def get_convex_hull(self, points):
        '''
        Compute the convex hull of given points
        :param points: a 2d-array of points
        :return: convexhull: a 2d-array of points making the convex hull of input points
        '''

        # http://scipy.github.io/devdocs/generated/scipy.spatial.ConvexHull.html
        hull = ConvexHull(points, False, 'QJ')

        conv_hull = list()

        for v in hull.vertices:
            conv_hull.append( points[v] )

        return np.array(conv_hull)

if __name__ == '__main__':

    from src.inkml import Inkml
    from src.helper import parse_raw_ink_data
    from src.data_preparation import preprocess
    import matplotlib.pyplot as plt

    # ink = Inkml('dataset/TrainINKML/HAMEX/formulaire024-equation050.inkml')
    ink = Inkml('dataset/myTraining/65_alfonso.inkml')

    parse_raw_ink_data(ink)
    preprocess(ink)

    # show how the expression looks like
    for s in ink.strokes:
        stroke = ink.strokes[s]
        plt.scatter(stroke[:, 0], stroke[:, 1], c='k', marker='.')

    los_graph = Graph(ink)

    # plotting a line between point pairs
    # https://stackoverflow.com/questions/35363444/plotting-lines-connecting-points?rq=1
    for n1 in los_graph.nodes:
        s1 = los_graph.nodes[n1]  # get node for s1

        print(s1.stroke_id, ':', s1.bb_center)

        for nbor in los_graph.edges[n1]:
            s2 = los_graph.nodes[nbor]  # get node for s2

            x1, y1 = s1.bb_center
            x2, y2 = s2.bb_center

            print('\t', s2.stroke_id, ':', s2.bb_center)

            plt.plot(x1, y1, 'ro')
            plt.plot(x2, y2, 'ro')
            # plt.plot([x1, x2], [y1, y2], 'b-')

    plt.gca().invert_yaxis()
    plt.show()

