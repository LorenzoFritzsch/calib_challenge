import numpy as np
import statistics
import math
import cv2
from matplotlib import pyplot as plt
import sys



def segment_length(xa, ya, xb, yb):
    return math.sqrt((xb - xa)**2 + (yb - ya)**2)

def apply_gaussian_filter(old_gray, gray):
    old_gray_blurred = cv2.GaussianBlur(old_gray, (5, 5), 0)
    gray_blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return old_gray_blurred, gray_blurred

def bilateral_filter(old_gray, gray):
    return cv2.bilateralFilter(old_gray, 9, 75, 75), cv2.bilateralFilter(gray, 9, 75, 75)

def get_standard_deviation(dataset):
    mean = np.average(dataset)
    n = len(dataset)
    return math.sqrt(sum((x - mean) ** 2 for x in dataset) / n)

def standardize(to_standardize, dataset):
    n = len(dataset)
    if n == 0:
        return to_standardize
    mean = sum(dataset) / n

    standard_deviation = math.sqrt(sum((x - mean) ** 2 for x in dataset) / n)
    if standard_deviation == 0.0:
        return
    standardized = (to_standardize - mean) / standard_deviation

    print("STANDARDIZING from " + str(n) + " elements. Mean: " + str(mean) + " Standard Deviation: " + str(standard_deviation))
    print("To standardize: " + str(to_standardize) + " Standardized: " + str(standardized))
    return standardized

def clear_dataset(dataset_x, dataset_y, max_x, max_y):
    x_without_nan_inf = remove_nan_or_inf_values_from_dataset(dataset_x)
    y_without_nan_inf = remove_nan_or_inf_values_from_dataset(dataset_y)

    x_in_bound, y_in_bound = remove_val_outside_bound(x_without_nan_inf, y_without_nan_inf, max_x, max_y)

    x_inside_standard_dev, y_inside_standard_dev = remove_val_outside_standard_dev(x_in_bound, y_in_bound)

    return x_inside_standard_dev, y_inside_standard_dev


def remove_nan_or_inf_values_from_dataset(dataset):
    for value in dataset:
        if math.isnan(value) or math.isinf(value):
            dataset.remove(value)
    return dataset

def remove_val_outside_standard_dev(dataset_x, dataset_y):

    standard_dev_x = get_standard_deviation(dataset_x)
    standard_dev_y = get_standard_deviation(dataset_y)

    len_dataset_x = len(dataset_x)
    i = 0

    while i in range(len_dataset_x):

        x = dataset_x[i]
        y = dataset_y[i]

        if x >= standard_dev_x or y >= standard_dev_y:
            dataset_x.pop(i)
            dataset_y.pop(i)

        i += 1
        len_dataset_x = len(dataset_x)

    return dataset_x, dataset_y

def remove_val_outside_bound(dataset_x, dataset_y, max_x, max_y):

    len_dataset_x = len(dataset_x)
    i = 0

    while i in range(len_dataset_x):

        x = dataset_x[i]
        y = dataset_y[i]

        if x > max_x or y > max_y:
            dataset_x.pop(i)
            dataset_y.pop(i)

        i += 1
        len_dataset_x = len(dataset_x)

    return dataset_x, dataset_y


def truncate(val):
    if math.isnan(val) or math.isinf(val):
        return val
    k = 10
    return int(val * k) / k

def cramer(m_one, c_one, d_one, m_two, c_two, d_two):
    q_one = m_one * c_one - d_one
    q_two = m_two * c_two - d_two

    matrix_determinant = [[m_one, 1], [m_two, 1]]
    matrix_x = [[q_one, 1], [q_two, 1]]
    matrix_y = [[m_one, q_one], [m_two, q_two]]

    #print(matrix_determinant, matrix_x, matrix_y)

    determinant = np.linalg.det(matrix_determinant)
    x_det = np.linalg.det(matrix_x)
    y_det = np.linalg.det(matrix_y)

    x_center = abs(x_det / determinant)
    y_center = abs(y_det / determinant)
    return truncate(x_center), truncate(y_center)

def actual_labeler(video_captured, x_center_all, y_center_all, mask, feature_params, lk_params, generation):
    focal_length_pixel = 910
    round = 1

    green = (20, 254, 87)
    red = (0, 0, 255)
    yellow = (255, 254, 52)

    # Get frame dimensions
    width = video_captured.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = video_captured.get(cv2.CAP_PROP_FRAME_HEIGHT)
    diagonal = math.sqrt(width**2 + height**2)
    fov = (2 * math.atan(diagonal / focal_length_pixel)) * 180 / math.pi

    ret, old_frame = video_captured.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

    car_area = [(0, int(height - 200)), (int(width), int(height - 200))]

    old_mask = mask

    lost_frames = 0
    frames_with_errors = 0
    max_frames = 1200

    while video_captured.isOpened():

        sys.stdout.write("\033[93m" + "\rFrame: %r" % round + " Generation: %r" % generation + "\033[0m")
        sys.stdout.flush()

        # Capture frame by frame
        ret, frame = video_captured.read()
        if lost_frames + round == max_frames:
            break
        if np.shape(frame) == ():
            lost_frames += 1
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply filters
        old_gray, gray = bilateral_filter(old_gray, gray)
        old_gray, gray = apply_gaussian_filter(old_gray, gray)

        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, gray, p0, None, **lk_params)

        # Select good points
        if p1 is not None:
            good_new = p1[st == 1]
            good_old = p0[st == 1]

            local_coords_x = []
            local_coords_y = []

            prev_m = None
            prev_c = None
            prev_d = None

            for i, (new, old) in enumerate(zip(good_new, good_old)):

                a, b = new.ravel()
                c, d = old.ravel()

                m = 0
                if a != c:
                    m = (d - b)/(c - a)
                else:
                    continue

                if b > car_area[0][1] or d > car_area[1][1]:
                    continue

                if prev_m == None:
                    prev_m = m
                    prev_c = c
                    prev_d = d

                x_center_current, y_center_current = cramer(prev_m, prev_c, prev_d, m, c, d)
                local_coords_x.append(x_center_current)
                local_coords_y.append(y_center_current)
                prev_m = m
                prev_c = c
                prev_d = d


            clean_local_coords_x, clean_local_coords_y = clear_dataset(local_coords_x, local_coords_y, width, height)

            x_center = np.average(clean_local_coords_x)
            y_center = np.average(clean_local_coords_y)

            x_in_bound = x_center < width
            y_in_bound = y_center < height
            coords_in_bounds = x_in_bound and y_in_bound


            x_center_all_average = np.average(x_center_all)
            y_center_all_average = np.average(y_center_all)


            if not math.isnan(x_center) and coords_in_bounds:

                if math.isnan(x_center_all_average):
                    x_center_all_average = x_center
                    y_center_all_average = y_center

                #Weighted average
                weight_xy_just_calculated = 1
                weight_xy_average = 2 * (generation + 1)

                x_center_wa = (weight_xy_just_calculated * x_center + weight_xy_average * x_center_all_average) / (weight_xy_just_calculated + weight_xy_average)
                y_center_wa = (weight_xy_just_calculated * y_center + weight_xy_average * y_center_all_average) / (weight_xy_just_calculated + weight_xy_average)


                x_center_all.append(x_center_wa)
                y_center_all.append(y_center_wa)

                frame = cv2.circle(frame, (int(x_center_wa), int(y_center_wa)), 5, red, -1)
                #frame = cv2.circle(frame, (int(x_center), int(y_center)), 4, yellow, -1)
                #frame = cv2.circle(frame, (630,  441), 5, red, -1) #from average
            else:
                frame = cv2.circle(frame, (int(x_center_all_average), int(y_center_all_average)), 5, red, -1)
                frames_with_errors += 1

            img = cv2.add(frame, mask)

            # Now update the previous frame and previous points
            old_gray = gray.copy()
            old_mask = mask
            p0 = good_new.reshape(-1, 1, 2)

            #Diagonals
            img = cv2.line(img, (0, 0), (int(width), int(height)), red)
            img = cv2.line(img, (int(width), 0), (0, int(height)), red)
            #Car, approx
            img = cv2.line(img, car_area[0], car_area[1], red)

            cv2.imshow('Main', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            round += 1

    video_captured.release()
    cv2.destroyAllWindows()

    return x_center_all, y_center_all, mask

class Labeler:
    x_center_all = []
    y_center_all = []

    video_number = 4

    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners=0, qualityLevel=0.01, minDistance=15, blockSize=5)

    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    video_captured = cv2.VideoCapture('../labeled/'+str(video_number)+'.hevc')

    ret, old_frame = video_captured.read()
    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)

    generation = 0

    while(True):
        video_captured = cv2.VideoCapture('../labeled/'+str(video_number)+'.hevc')
        x_center_all, y_center_all, mask = actual_labeler(video_captured, x_center_all, y_center_all, mask, feature_params, lk_params, generation)
        #Clear x_center_all and y_center_all
        x_center_all, y_center_all = remove_val_outside_standard_dev(x_center_all, y_center_all)
        print("\n Average x: " + str(np.average(x_center_all)) + " y: " +  str(np.average(y_center_all)))
        generation += 1
