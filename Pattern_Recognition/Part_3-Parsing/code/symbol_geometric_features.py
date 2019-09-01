'''
@author: Sanyukta Kate, Pratik Bongale
'''

import geometric_features

def features(s1_id, s2_id, seg_map, seg_strokes):
    # two symbols with their set of strokes is taken as input
    # find the geometric features

    s1_bb_center = seg_map[s1_id]
    s2_bb_center = seg_map[s2_id]

    # calculate average centers of the symbols
    stroke1_avg_centers = geometric_features.averaged_centers(seg_strokes[s1_id]["x"],seg_strokes[s1_id]["y"] )
    stroke2_avg_centers = geometric_features.averaged_centers(seg_strokes[s2_id]["x"],seg_strokes[s2_id]["y"])

    # 1. horizontal distance between the bounding_box_centers
    horizontal_distance = geometric_features.horizontal_distance_BB(s1_bb_center, s2_bb_center)
    #return [horizontal_distance]

    # 2. vertcial distance between the bounding_box_centers
    vertical_distance = geometric_features.vertical_distance_BB(s1_bb_center, s2_bb_center)

    # 3. distance_bounding_box_centers
    distance = geometric_features.distance_BB(s1_bb_center, s2_bb_center)

    # 4. distance average centers
    distance_avg_centers = geometric_features.distance_averaged_centers(stroke1_avg_centers, stroke2_avg_centers) # average values

    # 5. writing slope
    slope = geometric_features.writing_slope(seg_strokes[s1_id]["x"],seg_strokes[s1_id]["y"], seg_strokes[s2_id]["x"],seg_strokes[s2_id]["y"])

    # 6. maximal distance between two points of the symbol
    maximum_distance = geometric_features.maximal_distance(seg_strokes[s1_id]["x"],seg_strokes[s1_id]["y"], seg_strokes[s2_id]["x"],seg_strokes[s2_id]["y"])

    features = [horizontal_distance, vertical_distance, distance, distance_avg_centers, slope, maximum_distance]

    return features




