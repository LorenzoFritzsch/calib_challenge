import numpy as np
import statistics
import math
import cv2
import sys
import scipy.stats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
import tensorflow as tf

tf.__version__


np.seterr(divide='ignore', invalid='ignore')
np.set_printoptions(threshold=np.inf)

n_of_epochs = 100
n_of_neurons = 3
n_of_output = 1

n_of_feature_per_row = 5

focal_length_pixel = 910.0

video_number = 0
max_video_number = 4

# params for ShiTomasi corner detection
feature_params = dict(maxCorners=0, qualityLevel=0.01, minDistance=15, blockSize=5)

# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


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
    # Backward Elimination with standard deviation as significance level.

    max_val_x, min_val_x = get_max_min_value_considering_standard_dev(dataset_x)
    max_val_y, min_val_y = get_max_min_value_considering_standard_dev(dataset_y)

    len_dataset_x = len(dataset_x)

    cleared_dataset_x = []
    cleared_dataset_y = []

    for i in range(len_dataset_x):

        x = dataset_x[i]
        y = dataset_y[i]

        if max_val_x > x > min_val_x and max_val_y > y > min_val_y:
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

    for i in range(len(dataset_x)):  # dataset_x and dataset_y have the same size

        x = dataset_x[i]
        y = dataset_y[i]

        all_probabilities_x.append(scipy.stats.norm(average_x, stand_dev_x).pdf(x))
        all_probabilities_y.append(scipy.stats.norm(average_y, stand_dev_y).pdf(y))

    average_probability_x = np.average(all_probabilities_x)
    average_probability_y = np.average(all_probabilities_y)

    for i in range(len(all_probabilities_x)):  # all_probabilities_x and all_probabilities_y have the same size
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


def calculate_ang(center_direction, center_image, focal_length):

    distance = abs(center_image - center_direction)
    ang = math.atan(distance / focal_length)
    return ang


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
        cluster = KMeans(n_clusters=n_clusters)
        cluster_labels = cluster.fit_predict(dataToFit)
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
    most_populated_cluster = statistics.mode(clusters_list)

    items_in_most_populated_cluster = []

    for i in range(len(clusters_list)):
        cluster_number = clusters_list[i]
        if cluster_number == most_populated_cluster:
            items_in_most_populated_cluster.append(dataset[i])

    return np.average(items_in_most_populated_cluster)


def fix_x_y_centers(dataset_x, dataset_y):

    average_x = np.average(dataset_x)
    average_y = np.average(dataset_y)

    standard_dev_x = get_standard_deviation(dataset_x)
    standard_dev_y = get_standard_deviation(dataset_y)

    x_max = average_x + standard_dev_x
    x_min = average_x - standard_dev_x

    y_max = average_y + standard_dev_y
    y_min = average_y - standard_dev_y

    for i in range(len(dataset_x)):  # dataset_x and dataset_y have the same size

        x = dataset_x[i]
        y = dataset_y[i]

        x_outside_standard_dev = x > x_max or x < x_min
        y_outside_standard_dev = y > y_max or y < y_min

        if x_outside_standard_dev or y_outside_standard_dev:
            dataset_x[i] = average_x
            dataset_y[i] = average_y

    return dataset_x, dataset_y


def get_minimums_values(dataset, n_of_vals):

    coord_current_mean = np.average(dataset)
    # Get values nearest to avg
    distances = []
    for val in dataset:
        distance = abs(val - coord_current_mean)
        distances.append(distance)

    minimums = []

    for i in range(n_of_vals):
        min = np.argmin(distances)
        distances[min] = math.inf
        minimums.append(dataset[min])

    return minimums


def create_train_dataset(coords_per_frame, coords_average, center_image):
    # Create the train set
    train_set = []
    for i in range(len(coords_per_frame)):

        row = []
        coord_all = coords_per_frame[i]

        minimums = get_minimums_values(coord_all, 5)

        minimums_average = np.average(minimums)

        calculated_ang = calculate_ang(minimums_average, center_image, 910.0)
        distance = abs(center_image - minimums_average)

        row.append(minimums_average)
        row.append(coords_average)
        row.append(distance)
        row.append(910.0)
        row.append(center_image)
        row = np.array(row).reshape(1, n_of_feature_per_row)
        train_set.append(row)

    return train_set


def get_x_y_per_frame_and_average(dataset_x, dataset_y):

    x_per_frame = []
    y_per_frame = []

    x_very_all_for_avg = []
    y_very_all_for_avg = []

    x_statistically_viable, y_statistically_viable = get_only_statistically_viable_coords(dataset_x, dataset_y)
    x_in_standard_dev, y_in_standard_dev = remove_val_outside_standard_dev(x_statistically_viable, y_statistically_viable)

    for i in range(int(len(x_in_standard_dev))):

        sys.stdout.write("\rCalculating [] %r" % i + " / %r" % len(x_in_standard_dev))
        sys.stdout.flush()

        x_per_frame.append(x_in_standard_dev)
        x_very_all_for_avg += x_in_standard_dev

        y_per_frame.append(y_in_standard_dev)
        y_very_all_for_avg += y_in_standard_dev

    x_all_average = np.average(x_very_all_for_avg)
    y_all_average = np.average(y_very_all_for_avg)

    return x_per_frame, x_all_average, y_per_frame, y_all_average


def train_ann(train_set_x, yaws, train_set_y, pitches):
    ann_pitch = tf.keras.models.Sequential()
    ann_pitch.add(tf.keras.layers.Dense(units=n_of_neurons, activation='relu'))
    ann_pitch.add(tf.keras.layers.Dense(units=n_of_neurons, activation='relu'))
    ann_pitch.add(tf.keras.layers.Dense(units=n_of_output, activation='sigmoid'))
    ann_pitch.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    ann_yaw = tf.keras.models.Sequential()
    ann_yaw.add(tf.keras.layers.Dense(units=n_of_neurons, activation='relu'))
    ann_yaw.add(tf.keras.layers.Dense(units=n_of_neurons, activation='relu'))
    ann_yaw.add(tf.keras.layers.Dense(units=n_of_output, activation='sigmoid'))
    ann_yaw.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    ann_pitch.fit(train_set_y, pitches, batch_size=32, epochs=n_of_epochs)
    ann_yaw.fit(train_set_x, yaws, batch_size=32, epochs=n_of_epochs)

    return ann_pitch, ann_yaw


def get_x_y_dataset_from_dict(dict_frames):

    x_saved = []
    y_saved = []

    for i in range(int(len(dict_frames) / 2)):
        i += 1

        x_key = str(i) + "-x"
        y_key = str(i) + "-y"

        x_saved.append(dict_frames[x_key])
        y_saved.append(dict_frames[y_key])

    return x_saved, y_saved


def get_good_vals_from(x_local, y_local):
    x_good = []
    y_good = []

    x_all_list = []
    y_all_list = []

    for i in range(len(x_local)):
        #Get good values frame by frame

        sys.stdout.write("\rCalculating [GoodVals] %r" % (i+1) + " / %r" % len(x_local))
        sys.stdout.flush()

        x_current = x_local[i]
        y_current = y_local[i]

        x_statistically_viable, y_statistically_viable = get_only_statistically_viable_coords(x_current, y_current)
        x_in_standard_dev, y_in_standard_dev = remove_val_outside_standard_dev(x_statistically_viable, y_statistically_viable)

        x_good.append(np.average(x_in_standard_dev))
        y_good.append(np.average(y_in_standard_dev))

        x_all_list.extend(x_in_standard_dev)
        y_all_list.extend(y_in_standard_dev)

    return x_good, y_good, x_all_list, y_all_list


def train_ann_on_labeled_videos():

    n_of_labeled_videos = 4
    width = 0
    height = 0

    x_saved = []
    y_saved = []

    pitches = []
    yaws = []

    x_all_list = []
    y_all_list = []

    for video in range(n_of_labeled_videos):
        print("\nLearning from video " + str(video) + " of 4")
        video_captured = cv2.VideoCapture('../labeled/' + str(video) + '.hevc')

        width = video_captured.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = video_captured.get(cv2.CAP_PROP_FRAME_HEIGHT)

        dict_frames = get_center_of_direction_for_each_frame(video_captured, video)

        x_local, y_local = get_x_y_dataset_from_dict(dict_frames) # Returns 2d array of len n of frames

        x_good, y_good, x_all, y_all = get_good_vals_from(x_local, y_local)

        x_saved.extend(x_good)
        y_saved.extend(y_good)
        x_all_list.extend(x_all)
        y_all_list.extend(y_all)

        # Load labeled pitches and yaws
        pitches_yaws = np.loadtxt('../labeled/' + str(video) + '.txt')
        for i in pitches_yaws:
            pitches.append(i[0])
            yaws.append([i[1]])

    x_saved = np.array(x_saved).reshape(-1, 1)
    y_saved = np.array(y_saved).reshape(-1, 1)

    x_average_all = np.average(x_all_list)
    y_average_all = np.average(y_all_list)

    # Create the train set

    x_center_image = width / 2
    y_center_image = height / 2

    train_set_x = create_train_dataset(x_saved, x_average_all, x_center_image)
    train_set_y = create_train_dataset(y_saved, y_average_all, y_center_image)

    train_set_x = (np.array(train_set_x).reshape(-1, n_of_feature_per_row))
    train_set_y = (np.array(train_set_y).reshape(-1, n_of_feature_per_row))

    pitches = np.array(pitches).reshape(-1, 1).astype('float32')
    yaws = np.array(yaws).reshape(-1, 1).astype('float32')

    print(len(train_set_x), len(train_set_y), len(pitches), len(yaws))

    ann_pitch, ann_yaw = train_ann(train_set_x, yaws, train_set_y, pitches)

    return ann_pitch, ann_yaw


def get_center_of_direction_for_each_frame(video_captured, video_number):
    scanning_frame = 1

    # Get frame dimensions
    width = video_captured.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = video_captured.get(cv2.CAP_PROP_FRAME_HEIGHT)

    frames = get_all_frames(video_captured)

    dict_frames = {}

    old_gray = cv2.cvtColor(frames[1], cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    centers = []

    for frame in frames:

        key_x = str(scanning_frame)+"-x"
        key_y = str(scanning_frame)+"-y"

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

                if prev_m is None:
                    prev_m = m
                    prev_c = c
                    prev_d = d

                x_center_current, y_center_current = cramer(prev_m, prev_c, prev_d, m, c, d)

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


class Labeler:

    # Train ANN
    ann_pitch, ann_yaw = train_ann_on_labeled_videos()

    for video_number in range(max_video_number):

        video_captured = cv2.VideoCapture('../labeled/'+str(video_number)+'.hevc')

        dict_frames = get_center_of_direction_for_each_frame(video_captured, video_number)

        # Create the train set
        width = video_captured.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = video_captured.get(cv2.CAP_PROP_FRAME_HEIGHT)

        x_center_image = width / 2
        y_center_image = height / 2

        # Predict
        pitch_yaw_predicted = []
        for i in range(int(len(dict_frames)/2)):

            i += 1

            if i > int(len(dict_frames)/2):
                break

            sys.stdout.write("\rPredicting %r" % i + " / %r" % (len(dict_frames)/2))
            sys.stdout.flush()

            x_key = str(i) + "-x"
            y_key = str(i) + "-y"

            x_saved = dict_frames[x_key]
            y_saved = dict_frames[y_key]

            x_statistically_viable, y_statistically_viable = get_only_statistically_viable_coords(x_saved, y_saved)
            x_in_standard_dev, y_in_standard_dev = remove_val_outside_standard_dev(x_statistically_viable, y_statistically_viable)

            minimums_x = get_minimums_values(x_in_standard_dev, 5)
            minimums_y = get_minimums_values(y_in_standard_dev, 5)

            minimums_x_avg = np.average(minimums_x)
            minimums_y_avg = np.average(minimums_y)

            x_calc = np.average(x_in_standard_dev)
            y_calc = np.average(y_in_standard_dev)

            distance_x = abs(x_center_image - x_calc)
            distance_y = abs(y_center_image - y_calc)

            predicted_pitch = ann_pitch.predict(np.array([y_calc, minimums_y_avg, distance_x, 910.0, y_center_image]).reshape(1, n_of_feature_per_row))
            predicted_yaw = ann_yaw.predict(np.array([x_calc, minimums_x_avg, distance_y, 910.0, x_center_image]).reshape(1, n_of_feature_per_row))

            pitch_yaw_predicted.append([predicted_pitch, predicted_yaw])

        f = open(str(video_number)+".txt", "w")
        f.write(str(np.array(pitch_yaw_predicted).reshape(-1, 2)).replace("[", "").replace("]", ""))
        f.close()

        dict_frames = {}
