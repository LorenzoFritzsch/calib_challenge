import numpy as np
import statistics
import math
import cv2
import sys
import scipy.stats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering

import matplotlib.pyplot as plt

np.seterr(divide='ignore', invalid='ignore')

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

def get_only_statistically_viable_coords(dataset_x, dataset_y):
    average_x = np.average(dataset_x)
    average_y = np.average(dataset_y)

    stand_dev_x = get_standard_deviation(dataset_x)
    stand_dev_y = get_standard_deviation(dataset_y)

    viable_dataset_x = []
    viable_dataset_y = []

    all_probabilities_x = []
    all_probabilities_y = []

    len_dataset_x = len(dataset_x)
    len_dataset_y = len(dataset_y)

    for i in range(len(dataset_x)): #dataset_x and dataset_y have the same size

        x = dataset_x[i]
        y = dataset_y[i]

        all_probabilities_x.append(scipy.stats.norm(average_x, stand_dev_x).pdf(x))
        all_probabilities_y.append(scipy.stats.norm(average_y, stand_dev_y).pdf(y))

    average_probability_x = np.average(all_probabilities_x)
    average_probability_y = np.average(all_probabilities_y)

    for i in range(len(all_probabilities_x)): #all_probabilities_x and all_probabilities_y have the same size
        probability_x = all_probabilities_x[i]
        probability_y = all_probabilities_y[i]

        if probability_x >= average_probability_x and probability_y >= average_probability_y:
            viable_dataset_x.append(dataset_x[i])
            viable_dataset_y.append(dataset_y[i])

    return viable_dataset_x, viable_dataset_y


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

def get_k_for_kmeans(dataset):

    range_n_clusters = range(2, 5)
    dataToFit = np.array(dataset).reshape(-1, 1)
    best_clusters = 0
    previous_silh_avg = 0.0

    for n_clusters in range_n_clusters:
        clusterer = KMeans(n_clusters=n_clusters)
        cluster_labels = clusterer.fit_predict(dataToFit)
        silhouette_avg = silhouette_score(dataToFit, cluster_labels)

        if silhouette_avg > previous_silh_avg:
            previous_silh_avg = silhouette_avg
            best_clusters = n_clusters

    return best_clusters

def hierarchical_clustering(dataset):

    if len(dataset) <= 4:
        return np.average(dataset)

    number_of_clusters = get_k_for_kmeans(dataset)

    dataset = np.array(dataset).reshape(-1, 1)
    clustering = AgglomerativeClustering(n_clusters=number_of_clusters, affinity='euclidean', linkage='ward').fit(dataset)
    clusters_list = clustering.labels_
    most_populated_cluster =  statistics.mode(clusters_list)

    items_in_most_populated_cluster = []

    for i in range(len(clusters_list)):
        cluster_number = clusters_list[i]
        if cluster_number == most_populated_cluster:
            items_in_most_populated_cluster.append(dataset[i])

    return np.average(items_in_most_populated_cluster)


def get_center_of_direction_for_each_frame(video_captured, dict_frames, x_all, y_all, feature_params, lk_params, video_number, epoch):
    round = 1

    # Get frame dimensions
    width = video_captured.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = video_captured.get(cv2.CAP_PROP_FRAME_HEIGHT)


    frames = get_all_frames(video_captured)

    #ret, old_frame = video_captured.read()
    old_gray = cv2.cvtColor(frames[1], cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

    centers = []

    for frame in frames:

        key_x = str(round)+"-x"
        key_y = str(round)+"-y"

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

                if prev_m == None:
                    prev_m = m
                    prev_c = c
                    prev_d = d

                x_center_current, y_center_current = cramer(prev_m, prev_c, prev_d, m, c, d)

                are_nan_or_inf = math.isnan(x_center_current) and math.isnan(y_center_current) and math.isinf(x_center_current) and math.isinf(y_center_current)
                are_inside_bounds = x_center_current < width and y_center_current < height

                if not are_nan_or_inf and are_inside_bounds:
                    local_x.append(x_center_current)
                    local_y.append(y_center_current)

                    x_all.append(x_center_current)
                    y_all.append(y_center_current)


                prev_m = m
                prev_c = c
                prev_d = d

            if epoch > 0:
                local_x += dict_frames[key_x]
                local_y += dict_frames[key_y]

            dict_frames.update({key_x: local_x})
            dict_frames.update({key_y: local_y})

            # Now update the previous frame and previous points
            old_gray = gray.copy()
            p0 = good_new.reshape(-1, 1, 2)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            sys.stdout.write("\033[93m" + "\rVideo: %r" % video_number + " Frame: %r" % round + " Epoch: %r" % epoch + "\033[0m")
            sys.stdout.flush()

            round += 1

    video_captured.release()
    cv2.destroyAllWindows()

    return dict_frames, x_all, y_all

class Labeler:

    focal_length_pixel = 910

    x_center_all = []
    y_center_all = []

    video_number = 0
    max_video_number = 0

    n_of_epochs = 1

    dict_frames = {}

    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners=0, qualityLevel=0.01, minDistance=15, blockSize=5)

    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    epoch = 0

    while(True):

        if video_number > max_video_number:
            break

        video_captured = cv2.VideoCapture('../labeled/'+str(video_number)+'.hevc')
        width = video_captured.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = video_captured.get(cv2.CAP_PROP_FRAME_HEIGHT)

        dict_frames, x_center_all, y_center_all = get_center_of_direction_for_each_frame(video_captured, dict_frames, x_center_all, y_center_all, feature_params, lk_params, video_number, epoch)

        if epoch == n_of_epochs:

            pitch_yaw = []

            print("\n")

            for i in range(int(len(dict_frames)/2)):

                i += 1

                sys.stdout.write("\rCalulating pitch and yaw angles, frame: %r" % i + " / %r " % int(len(dict_frames) / 2))
                sys.stdout.flush()

                x_key = str(i) + "-x"
                y_key = str(i) + "-y"

                #BETTER CLEAN UP, CONSIDER COUPLE OF COORDINATES, NOT ONLY X, Y SEPARATELY

                # Add all the other values
                x_saved = dict_frames[x_key] + x_center_all
                y_saved = dict_frames[y_key] + y_center_all

                x_statistically_viable, y_statistically_viable = get_only_statistically_viable_coords(x_saved, y_saved)
                x_in_standard_dev, y_in_standard_dev = remove_val_outside_standard_dev(x_statistically_viable, y_statistically_viable)

                """
                if i < 5:
                    plt.scatter(x_in_standard_dev, y_in_standard_dev)
                    plt.title(i)
                    plt.show()
                    #plt.close()
                """

                x = hierarchical_clustering(x_in_standard_dev)
                y = hierarchical_clustering(y_in_standard_dev)

                #x = np.average(x_in_standard_dev)
                #y = np.average(y_in_standard_dev)

                pitch, yaw = calculate_pitch_and_yaw([x, y], [width/2, height/2], focal_length_pixel)
                pitch_yaw.append([pitch, yaw])

            print("\n")

            pitch_yaw_reshaped = str(pitch_yaw).replace("], [", "\n").replace(", ", " ").replace("[[", "").replace("]]", "")

            f = open(str(video_number)+".txt", "w")
            f.write(str(pitch_yaw_reshaped))
            f.close()


            epoch = -1
            video_number += 1
            x_center_all = []
            y_center_all = []
            dict_frames = {}

        epoch += 1
