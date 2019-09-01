import numpy as np

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


if __name__ == '__main__':

    print(parse_raw_stroke('1 2, 4 5, 10 15'))
    print(parse_raw_stroke('1 2 198, 4 5 3847, 10 15 192827'))