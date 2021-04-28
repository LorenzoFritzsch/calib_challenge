import cv2

import sys

import numpy as np
import math

from filters import filters
from params import params

class get_center_of_direction:

    
    # params for ShiTomasi corner detection
    feature_params = params.feature_params

    # Parameters for lucas kanade optical flow
    lk_params = params.lk_params

    
    def get_center_of_direction_for_each_frame(video_captured, video_number):
        scanning_frame = 1

        # Get frame dimensions
        width = video_captured.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = video_captured.get(cv2.CAP_PROP_FRAME_HEIGHT)

        frames = get_center_of_direction.get_all_frames(video_captured)

        dict_frames = {}

        old_gray = cv2.cvtColor(frames[1], cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **get_center_of_direction.feature_params)

        centers = []

        for frame in frames:

            key_x = str(scanning_frame)+"-x"
            key_y = str(scanning_frame)+"-y"

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Apply filters
            old_gray, gray = filters.bilateral_filter(old_gray, gray)
            old_gray, gray = filters.gaussian_filter(old_gray, gray)

            p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **get_center_of_direction.feature_params)

            # calculate optical flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, gray, p0, None, **get_center_of_direction.lk_params)

            # Select good points
            if p1 is not None:
                good_new = p1[st == 1]
                good_old = p0[st == 1]

                prev_m = None
                prev_c = None
                prev_d = None

                local_x = []
                local_y = []

                for i, (new, old) in enumerate(zip(good_new, good_old)):

                    a, b = new.ravel()
                    c, d = old.ravel()

                    m = 0
                    if a != c:
                        m = (d - b)/(c - a)
                    else:
                        continue

                    if prev_m is None:
                        prev_m = m
                        prev_c = c
                        prev_d = d

                    x_center_current, y_center_current = get_center_of_direction.__cramer(prev_m, prev_c, prev_d, m, c, d)

                    are_nan_or_inf = math.isnan(x_center_current) and math.isnan(y_center_current) and math.isinf(x_center_current) and math.isinf(y_center_current)
                    are_inside_bounds = x_center_current < width and y_center_current < height

                    if not are_nan_or_inf and are_inside_bounds:
                        local_x.append(x_center_current)
                        local_y.append(y_center_current)

                    prev_m = m
                    prev_c = c
                    prev_d = d

                dict_frames.update({key_x: local_x})
                dict_frames.update({key_y: local_y})

                # Now update the previous frame and previous points
                old_gray = gray.copy()
                p0 = good_new.reshape(-1, 1, 2)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                sys.stdout.write("\033[93m" + "\rVideo: %r" % video_number + " Frame: %r" % scanning_frame + "\033[0m")
                sys.stdout.flush()

                scanning_frame += 1

        return dict_frames


    def get_all_frames(video_captured):
        all_frames = []
        while video_captured.isOpened():
            ret, frame = video_captured.read()
            if not ret:
                break
            all_frames.append(frame)
        return all_frames


    
    def __cramer(m_one, c_one, d_one, m_two, c_two, d_two):
        q_one = m_one * c_one - d_one
        q_two = m_two * c_two - d_two

        matrix_determinant = [[m_one, 1], [m_two, 1]]
        matrix_x = [[q_one, 1], [q_two, 1]]
        matrix_y = [[m_one, q_one], [m_two, q_two]]

        determinant = np.linalg.det(matrix_determinant)
        x_det = np.linalg.det(matrix_x)
        y_det = np.linalg.det(matrix_y)

        x_center = abs(x_det / determinant)
        y_center = abs(y_det / determinant)
        return x_center, y_center


