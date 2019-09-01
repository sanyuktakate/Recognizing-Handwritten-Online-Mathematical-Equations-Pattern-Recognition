'''
@author: Sanyukta Kate, Pratik Bongale
'''

import numpy as np
import math

def parse_raw_ink_data(ink):
    '''
    Parse the raw string data present in an Inkml object to a numpy array
    :param ink: an Inkml object to be parsed
    :return: None
    '''

    for stroke in ink.strokes:
        ink.strokes[stroke] = parse_raw_stroke(ink.strokes[stroke])


def parse_raw_stroke(s):
    '''
    Process one raw stroke points and return a numpy array
    :param s: string of points(x y) in the form 'x1 y1, x2 y2, x3 y3 ...'
    :return: 2d-numpy array of points in given segment
    '''

    s = [a.strip() for a in s.split(",")]
    points = []

    for pt in s:

        pt = pt.split(" ")

        if len(pt) > 1:
            x, y = float(pt[0]), float(pt[1])
        else:
            x, y = 0.0, 0.0

        points.append([x, y])

    return np.array(points)

def read_lg(fname, ink_obj):
    '''
    Reads relationships from label graph file and return the ground truth relationships
    :param fname:
    :param ink_obj:
    :return: dictionary of the form { (seg_id_1, seg_id_2) : 'Right' }
    '''

    '''
    # IUD, 2013_IVC_CROHME_F2_E3
    # Objects(3):
    O, }_1, \}, 1.0, 3
    O, {_1, \{, 1.0, 0
    O, T_1, T, 1.0, 2, 1

    # Relations from SRT:
    R, T_1, }_1, Right, 1.0
    R, {_1, T_1, Right, 1.0
    '''

    # read objetcs and relationships from ground truth file
    ink_obj.segMap = dict()
    res = dict()

    with open(fname, 'r') as lg_file:
        for line in lg_file:
            if line.startswith("O"):
                line = [a.strip() for a in line.split(",")]
                strokes = sorted([int(a) for a in line[4:]])

                label_id = line[1]

                for s in ink_obj.segments:
                    seg = ink_obj.segments[s]

                    if set(line[4:]) - seg.strId:
                        # go to next segment
                        continue
                    else:
                        ink_obj.segMap[label_id] = seg.id
                        break

            elif line.startswith("R"):
                line = [a.strip() for a in line.split(",")]

                u = ink_obj.segMap[line[1]]     # get segment id of first symbol
                v = ink_obj.segMap[line[2]]     # get segment id of second symbol

                res[(u,v)] = line[3]

    return res

def find_leftmost_symb(ink):

    min_seg = ''
    min_seg_x = math.inf
    min_x = math.inf

    # find the segment with the minimum x co-ordinate stroke
    for seg_id in ink.segments:

        seg_strokes = ink.segments[seg_id].strId

        # find the minimum x point of all strokes in this segment
        for s in seg_strokes:
            stroke = ink.strokes[s]
            min_x = min(stroke[:,0].min(), min_x)

        if min_x < min_seg_x:
            min_seg_x = min_x
            min_seg = seg_id

    return min_seg

def make_sc(graph):
    '''
    convert graph into a fully connected graph
    '''
    # get all the vertices of the graph
    vertices = list(graph.keys())
    for v in graph:
        neighbors = graph[v]

        for i in vertices:
            if i not in neighbors and i != v:
                graph[v].append(i)

    return graph

if __name__ == '__main__':

    # print(parse_raw_stroke('1 2, 4 5, 10 15'))
    # print(parse_raw_stroke('1 2 198, 4 5 3847, 10 15 192827'))

    # ink = Inkml('dataset/myTraining/inkml/65_alfonso.inkml')
    # parse_raw_ink_data(ink)
    # print(find_leftmost_symb(ink))

    graph = {"a": ["b"], "b": ["c"], "c": []}

    print(make_sc(graph))