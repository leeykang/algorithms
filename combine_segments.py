import numpy as np

def generate_curves(line_segments, tolerance=1e-7):
    """
    Generates open and closed curves from a list of unordered line segments.

    The algorithm is as follows:
    1. Randomly selects a single line segment from the list of available line
    segments as the current main line segment.

    2. Remove the current main line segment from the list of available line
    segments.

    3. Select an endpoint from the current main line segment.

    4. Find a line segment from the list of available line segments with an
    endpoint that is equivalent to the selected main endpoint (within a certain
    tolerance).
        a. If such an endpoint is found, add its corresponding line segment to
        the main line segment, remove the corresponding line segment from the
        list of available line segments and assign the new main endpoint to be
        the other point in the corresponding line segment.

        b. If an endpoint is not found, the possibilities include:
            1) The main endpoint is equal to the other endpoint of the main line
            segment, in which case a closed curve is obtained and is added to
            curves_list.

            2) An open curve is obtained from the first endpoint of the current
            main line segment, in which case step 4 is repeated for the other
            endpoint in the current main line segment.

            3) An open curve is obtained from both endpoints of the current main
            line segment, in which case the open curve is added to curves_list.

            4) The endpoint is also in another curve, creating an incorrect
            curve. The incorrect endpoint is printed out and an empty
            curves_list is returned from the function.

    5. Repeat steps 1 to 4 until there are no available line segments remaining
    or step 4b4 is performed, during which curves_list is returned from the
    function.

    Parameter(s):
    line_segments: A list containing line segments, with each line segment
    represented as a list containing two points, each point represented as a
    list of n values, and each value corresponding to an axis (e.g. x-dimension
    or y-dimension or z-dimension). Alternatively, an ndarray of the exact same
    form, corresponding to an array shape of (num_line_segments, 2, num_dimensions).

    tolerance (optional, default 1e-7): Tolerance for determining whether or not
    two points are the same. If the maximum difference between two points for
    each dimension is below the tolerance for every axis, the two points are
    treated as the same point.

    Return(s):
    curves_list: A list of all connected curves (open or closed), assuming
    all line segments are valid, else an empty list.

    """
    # Initialises the return value.
    curves_list = []

    # Convert array_like input to array if required.
    line_segments_arr = np.asarray(line_segments)

    # Compute the number of dimensions for the points.
    num_dimensions = line_segments_arr.shape[-1]

    # Compute the number of points, using the fact that each line segment
    # consists of 2 points.
    num_points = line_segments_arr.size // 2

    # Places a point of every line segment in rows of the base_points array.
    base_points = line_segments_arr.reshape((num_points, num_dimensions))

    # Assigns an index to every point, where the ith line segment has points
    # with indexes 2i and 2i+1.
    full_base_points = np.concatenate([base_points, np.expand_dims(np.arange(num_points),axis=1)], axis=1)

    # Provides sorting indices based on (z-dimension), y-dimension then
    # x-dimension.
    ind_for_sorting = np.lexsort([base_points[:,idx] for idx in reversed(range(num_dimensions))])

    # Uses sorting indices to sort every point and its associated index.
    full_point_lexsort = full_base_points[ind_for_sorting]

    # Separate the point and indexes array into its coordinates and its index.
    sorted_points, sorted_idx = np.split(full_point_lexsort, [num_dimensions], axis=1)

    # Finds equivalent points (based on a specified tolerance) and obtains their
    # respective indexes.
    equal_points = np.nonzero((np.all(np.abs((sorted_points - np.roll(sorted_points,1,axis=0))) < tolerance , axis=1) * 1))
    equal_idx_pairs = np.concatenate([np.roll(sorted_idx,1,axis=0), sorted_idx], axis=1)[equal_points].astype(int).tolist()

    # Adds the equivalent point indexes into two dictionaries, which corresponds
    # to the idea that the line segments can be oriented in the original or
    # opposite manner.
    first_equal_dict = {c: v for c,v in equal_idx_pairs}
    second_equal_dict = {v: c for c,v in equal_idx_pairs}

    # Creates an endpoint dictionary for providing a list of available line
    # segments by tracking available endpoints and checking if there are
    # incorrect curves (since a used endpoint will no longer be in the
    # endpoint dictionary).
    end_point_dict = {c: 1 for c in range(num_points)}

    try:
        # Loops by checking for available endpoints.
        while len(end_point_dict) > 0:

            # Randomly selects a main line segment and removes an endpoint from
            # the main line segment from the dictionary (steps 1 and 2).
            current_first_endpoint, __ = end_point_dict.popitem()

            # Finds the other endpoint of the main line segment and removes it
            # from the dictionary, and selects an endpoint from the main line
            # segment (assumed to be the endpoint with the bigger index)
            # (steps 2 and 3).
            if current_first_endpoint % 2:
                current_first_endpoint, current_second_endpoint = current_first_endpoint - 1, current_first_endpoint
                del end_point_dict[current_first_endpoint]

            else:
                current_second_endpoint = current_first_endpoint + 1
                del end_point_dict[current_second_endpoint]

            # Properly initialises the main line segment by placing the endpoint
            # with the bigger index on the right.
            current_segment = [current_first_endpoint, current_second_endpoint]

            # Assumes all curves are open and a check on the left endpoint is
            # required.
            skip_left_check = False

            # Perform check on right endpoint.
            while True:
                # Checks for right endpoint in both equivalent point index
                # dictionaries.
                right_first_check = current_second_endpoint in first_equal_dict
                right_second_check = current_second_endpoint in second_equal_dict

                if right_first_check or right_second_check:
                    # Remove the equivalent point index relationship from both
                    # dictionaries and obtain the two equivalent points.
                    if right_first_check:
                        corresponding_value = first_equal_dict[current_second_endpoint]
                        del first_equal_dict[current_second_endpoint]
                        del second_equal_dict[corresponding_value]

                    else:
                        corresponding_value = second_equal_dict[current_second_endpoint]
                        del second_equal_dict[current_second_endpoint]
                        del first_equal_dict[corresponding_value]

                    if corresponding_value == current_first_endpoint:
                        # Handles case where curve is closed.
                        skip_left_check = True
                        break

                    else:
                        # Remove corresponding index as an available starting
                        # point.
                        del end_point_dict[corresponding_value]

                    # Find the corresponding index line segment's other
                    # endpoint.
                    if corresponding_value % 2:
                        final_value = corresponding_value - 1

                    else:
                        final_value = corresponding_value + 1

                    # Add the corresponding index line segment's other
                    # endpoint to the main line segment.
                    current_segment.append(final_value)

                    if final_value == current_first_endpoint:
                        # Handles case where curve is closed.
                        skip_left_check = True
                        break

                    else:
                        # Remove corresponding index line segment's other
                        # endpoint as an available starting point.
                        del end_point_dict[final_value]

                        # Set the new right endpoint to be the corresponding
                        # index line segment's other endpoint.
                        current_second_endpoint = final_value

                else:
                    # Handles case where the curve is open.
                    break

            if not skip_left_check:
                # Only perform check on left endpoint if curve is open.
                while True:
                    # Checks for left endpoint in both equivalent point index
                    # dictionaries.
                    left_first_check = current_first_endpoint in first_equal_dict
                    left_second_check = current_first_endpoint in second_equal_dict

                    if left_first_check or left_second_check:
                        # Remove the equivalent point index relationship from both
                        # dictionaries and obtain the two equivalent points.
                        if left_first_check:
                            corresponding_value = first_equal_dict[current_first_endpoint]
                            del first_equal_dict[current_first_endpoint]
                            del second_equal_dict[corresponding_value]

                        else:
                            corresponding_value = second_equal_dict[current_first_endpoint]
                            del second_equal_dict[current_first_endpoint]
                            del first_equal_dict[corresponding_value]

                        # Remove corresponding index as an available starting
                        # point.
                        del end_point_dict[corresponding_value]

                        # Find the corresponding index line segment's other
                        # endpoint.
                        if corresponding_value % 2:
                            final_value = corresponding_value - 1

                        else:
                            final_value = corresponding_value + 1

                        # Add the corresponding index line segment's other
                        # endpoint to the main line segment.
                        current_segment.append(final_value)

                        # Remove corresponding index line segment's other
                        # endpoint as an available starting point.
                        del end_point_dict[final_value]

                        # Set the new left endpoint to be the corresponding
                        # index line segment's other endpoint.
                        current_first_endpoint = final_value

                    else:
                        # Handles case where the curve can no longer be
                        # extended.
                        break

            # Adds the coordinates of the main line segment to curves_list.
            curves_list.append(base_points[current_segment])

    # Catches the KeyError caused by an endpoint which creates an incorrect
    # curve.
    except KeyError as e:
        print('%s is found in two distinct curves.' % base_points[int(str(e))])
        curves_list = []

    return curves_list

if __name__ == '__main__':
    # Runs an example.
    contour_arr = np.array([[[1,1],[2,2]],[[3,3],[4,4]],[[6,6],[7,7]],[[6,6],[5,5]],[[7,7],[8,8]],[[2,2],[3,3]],[[4,4],[5,5]],[[1,1],[8,8]],[[9,9],[10,10]],[[11,11],[10,10]],[[0,0],[9,9]],[[11,11],[12,12]]])
    print(generate_curves(contour_arr))
