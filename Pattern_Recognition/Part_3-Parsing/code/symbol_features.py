'''
@author: Sanyukta Kate, Pratik Bongale
'''

# calls the symbol features which is the
import geometric_features
import symbol_geometric_features
import symbol_shape_features

def get_symb_features(s1, s2, ink):

    # do the processing of the symbols s1 and s2

    # segments/symbols present in the ink expression
    segments = ink.segments

    features = list()

    # create seg map
    seg_material = build_seg_materials(segments, ink)
    # print(seg_material)
    # seg_material [0] has the seg_map and seg_material[1] will have seg_strokes
    geo_feature_vec = symbol_geometric_features.features(s1, s2, seg_material[0], seg_material[1])
    parzen_feature_vec = symbol_shape_features.parzen_features(s1, s2, seg_material[0], seg_material[1])

    features.append(geo_feature_vec + parzen_feature_vec)
    return features


def build_seg_materials(segments, ink):
    seg_map = dict()
    seg_strokes = dict()

    for s in segments:  # take the first segment from the segment objects of an expression
        seg_stroke_ids = segments[s].strId
        x_list = []
        y_list = []
        for stroke_id in seg_stroke_ids:
            stroke = ink.strokes[
                stroke_id]  # This is one stroke from the segemnt. Keep accessing every stroke from the same segment
            # Now, get the x and y coordinates of this stroke
            stroke_x_list, stroke_y_list = stroke[:, 0], stroke[:, 1]
            # x_list and y_list will have all the points from the symbol
            x_list.extend(stroke_x_list)
            y_list.extend(stroke_y_list)

            bb_center_point = geometric_features.bounding_box(x_list, y_list)
            seg_map[s] = bb_center_point
            # store the segment's all strokes together in seg_strokes as a dictionary,
            # {"segment1": {"x":[], "y":[]}, "segment2":{"x":..}}
            seg_strokes[s] = dict()
            seg_strokes[s]["x"] = x_list
            seg_strokes[s]["y"] = y_list

    return [seg_map, seg_strokes]

