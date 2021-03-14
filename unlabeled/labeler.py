import numpy as np
import statistics
import math
import cv2
import sys
import scipy.stats


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


def remove_nan_or_inf_values_from_dataset(dataset):
    dataset_cleared = []
    for value in dataset:
        if not math.isnan(value) and not math.isinf(value):
            dataset_cleared.append(value)
    return dataset_cleared

def remove_val_outside_standard_dev(dataset_x, dataset_y):
    #Backward Elimination with standard deviation as significance level.

    max_val_x, min_val_x = get_max_min_value_considering_standard_dev(dataset_x)
    max_val_y, min_val_y = get_max_min_value_considering_standard_dev(dataset_y)

    len_dataset_x = len(dataset_x)

    cleared_dataset_x = []
    cleared_dataset_y = []

    for i in range(len_dataset_x):

        x = dataset_x[i]
        y = dataset_y[i]

        if x < max_val_x and y < max_val_y and x > min_val_x and y > min_val_y:
            cleared_dataset_x.append(x)
            cleared_dataset_y.append(y)

    return cleared_dataset_x, cleared_dataset_y

def get_only_statistically_viable_coords(dataset):
    average = np.average(dataset)
    stand_dev = get_standard_deviation(dataset)

    viable_dataset = []
    all_probabilities = []

    index_one = 0
    len_dataset = len(dataset)

    print("\n")

    for i in dataset:
        all_probabilities.append(scipy.stats.norm(average,stand_dev).pdf(i))

        index_one += 1
        progress = int((index_one / len_dataset) * 100)
        sys.stdout.write("\r1/2 Getting statistically viable coords, progress: %r" % progress + "%")
        sys.stdout.flush()

    average_probability = np.average(all_probabilities)

    print("\n")

    index_two = 0

    for i in range(len(all_probabilities)):
        probability = all_probabilities[i]#scipy.stats.norm(average,stand_dev).pdf(i)
        if probability >= average_probability:
            viable_dataset.append(dataset[i])

        index_two += 1
        progress = int((index_two / len_dataset) * 100)
        sys.stdout.write("\r2/2 Getting statistically viable coords, progress: %r" % progress + "%")
        sys.stdout.flush()

    return viable_dataset


def get_max_min_value_considering_standard_dev(dataset):

    average = np.average(dataset)

    standard_dev = get_standard_deviation(dataset)

    max_val = average + standard_dev
    min_val = average - standard_dev

    return max_val, min_val

def remove_val_outside_bound(dataset_x, dataset_y, max_x, max_y):

    len_dataset_x = len(dataset_x)

    dataset_cleared_x = []
    dataset_cleared_y = []

    for i in range(len_dataset_x):
        x = dataset_x[i]
        y = dataset_y[i]

        if x < max_x and y < max_y:
            dataset_cleared_x.append(x)
            dataset_cleared_y.append(y)

    return dataset_cleared_x, dataset_cleared_y

def cramer(m_one, c_one, d_one, m_two, c_two, d_two):
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

def calculate_pitch_and_yaw(center_direction, center_image, focal_length):
    x_center_direction = center_direction[0]
    y_center_direction = center_direction[1]

    x_center_image = center_image[0]
    y_center_image = center_image[1]

    y_distance = abs(y_center_image - y_center_direction)
    x_distance = abs(x_center_image - x_center_direction)

    pitch = math.atan(y_distance / focal_length)
    yaw = math.atan(x_distance / focal_length)

    return pitch, yaw

def get_all_frames(video_captured):
    all_frames = []
    while video_captured.isOpened():

        ret, frame = video_captured.read()
        if not ret:
            break
        all_frames.append(frame)
    return all_frames

def actual_labeler(video_captured, x_center_all, y_center_all, pitch_yaw_old, feature_params, lk_params, epoch):
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

    frames = get_all_frames(video_captured)

    #ret, old_frame = video_captured.read()
    old_gray = cv2.cvtColor(frames[1], cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)


    car_area = [(0, int(height - 200)), (int(width), int(height - 200))]

    pitch_yaw = []

    for frame in frames:

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

                are_nan_or_inf = math.isnan(x_center_current) and math.isnan(y_center_current) and math.isinf(x_center_current) and math.isinf(y_center_current)
                are_inside_bounds = x_center_current < width and y_center_current < height

                if not are_nan_or_inf and are_inside_bounds:
                    x_center_all.append(x_center_current)
                    y_center_all.append(y_center_current)


                prev_m = m
                prev_c = c
                prev_d = d

            clear_x_all, clear_y_all = remove_val_outside_standard_dev(x_center_all, y_center_all)


            #Use gaussian distribution
            x_center = np.average(clear_x_all)
            y_center = np.average(clear_y_all)


            pitch, yaw = calculate_pitch_and_yaw([x_center, y_center], [width/2, height/2], focal_length_pixel)

            if epoch != 0:
                old_pitch_yaw = pitch_yaw_old[round - 1]
                pitch_average = (old_pitch_yaw[0] + pitch) / 2
                yaw_average = (old_pitch_yaw[1] + yaw) / 2
                pitch_yaw.append([pitch_average, yaw_average])
            else:
                pitch_yaw.append([pitch, yaw])


            # Now update the previous frame and previous points
            old_gray = gray.copy()
            p0 = good_new.reshape(-1, 1, 2)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            sys.stdout.write("\033[93m" + "\rFrame: %r" % round + " Epoch: %r" % epoch + "\033[0m")
            sys.stdout.flush()

            round += 1

    video_captured.release()
    cv2.destroyAllWindows()

    return x_center_all, y_center_all, pitch_yaw

class Labeler:
    x_center_all = []
    y_center_all = []

    old_pitch_yaw = []

    video_number = 0
    max_video_number = 5

    n_of_epochs = 2

    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners=0, qualityLevel=0.01, minDistance=15, blockSize=5)

    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    epoch = 0

    while(True):
        video_captured = cv2.VideoCapture('../labeled/'+str(video_number)+'.hevc')
        print("\nVideo number: " + str(video_number))
        x_center_all, y_center_all, pitch_yaw = actual_labeler(video_captured, x_center_all, y_center_all, old_pitch_yaw, feature_params, lk_params, epoch)
        #Clear x_center_all and y_center_all
        x_center_all, y_center_all = remove_val_outside_standard_dev(x_center_all, y_center_all)
        x_center_all = remove_nan_or_inf_values_from_dataset(x_center_all)
        y_center_all = remove_nan_or_inf_values_from_dataset(y_center_all)

        x_center_all = get_only_statistically_viable_coords(x_center_all)
        y_center_all = get_only_statistically_viable_coords(y_center_all)

        old_pitch_yaw = pitch_yaw

        pitch_yaw_reshaped = str(pitch_yaw).replace("], [", "\n").replace(", ", " ").replace("[[", "").replace("]]", "")

        f = open(str(video_number)+".txt", "w")
        f.write(str(pitch_yaw_reshaped))
        f.close()

        if video_number == max_video_number:
            break

        if epoch == n_of_epochs:
            epoch = -1
            video_number += 1
            x_center_all = []
            y_center_all = []

        epoch += 1
