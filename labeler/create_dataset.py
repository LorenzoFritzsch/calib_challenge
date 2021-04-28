import numpy as np
import math

import cv2
import sys

from statistic_utils import statistic_utils
from get_center_of_direction import get_center_of_direction
from params import params

np.seterr(divide='ignore', invalid='ignore')
np.set_printoptions(threshold=np.inf)


class create_dataset:
    
    """
    Parameters
    """

    n_of_feature_per_row = params.n_of_features_per_row

    # Paths
    root_dir_labeled = params.root_dir_labeled
    root_dir_dataset = params.root_dir_dataset

    def __init__(self):
        pass


    def create_dataset_using_labeled_videos():

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
            video_captured = cv2.VideoCapture(create_dataset.root_dir_labeled + str(video) + '.hevc')

            width = video_captured.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = video_captured.get(cv2.CAP_PROP_FRAME_HEIGHT)

            dict_frames = get_center_of_direction.get_center_of_direction_for_each_frame(video_captured, video)

            x_local, y_local = create_dataset.__get_x_y_dataset_from_dict(dict_frames) # Returns 2d array of len n of frames

            x_good, y_good, x_all, y_all = create_dataset.__get_good_vals(x_local, y_local)

            x_saved.extend(x_good)
            y_saved.extend(y_good)
            x_all_list.extend(x_all)
            y_all_list.extend(y_all)

            # Load labeled pitches and yaws
            pitches_yaws = np.loadtxt(create_dataset.root_dir_labeled + str(video) + '.txt')
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

        train_set_x = create_dataset.__create_train_dataset(x_saved, x_average_all, x_center_image)
        train_set_y = create_dataset.__create_train_dataset(y_saved, y_average_all, y_center_image)

        train_set_cleared = []
        yaws_pitches_cleared = []

        # Replace nan with average
        average_yaw = np.nanmean(yaws)
        average_pitch = np.nanmean(pitches)
        for i in range(len(yaws)):
            yaw = yaws[i][0]
            pitch = pitches[i]
            if math.isnan(yaw):
                yaw = average_yaw
            if math.isnan(pitch):
                pitch = average_pitch
            
            train_set_cleared.append(train_set_x[i])
            train_set_cleared.append(train_set_y[i])
            yaws_pitches_cleared.append(yaw)
            yaws_pitches_cleared.append(pitch)
            
        train_set_cleared = np.array(train_set_cleared).reshape(-1, create_dataset.n_of_feature_per_row)
        yaws_pitches_cleared = np.array(yaws_pitches_cleared).reshape(-1, 1).astype('float32')
        
        np.save(create_dataset.root_dir_dataset + 'train_set.npy', train_set_cleared)
        np.save(create_dataset.root_dir_dataset + 'yaws_pitches.npy', yaws_pitches_cleared)

    """
    Methods for modify datasets
    """

    def __get_x_y_dataset_from_dict(dict_frames):

        x_saved = []
        y_saved = []

        for i in range(int(len(dict_frames) / 2)):
            i += 1

            x_key = str(i) + "-x"
            y_key = str(i) + "-y"

            x_saved.append(dict_frames[x_key])
            y_saved.append(dict_frames[y_key])

        return x_saved, y_saved


    def __get_good_vals(x_local, y_local):
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

            x_statistically_viable, y_statistically_viable = statistic_utils.get_only_statistically_viable_coords(x_current, y_current)
            x_in_standard_dev, y_in_standard_dev = statistic_utils.remove_val_outside_standard_dev(x_statistically_viable, y_statistically_viable)

            x_good.append(np.average(x_in_standard_dev))
            y_good.append(np.average(y_in_standard_dev))

            x_all_list.extend(x_in_standard_dev)
            y_all_list.extend(y_in_standard_dev)

        return x_good, y_good, x_all_list, y_all_list


    def __create_train_dataset(coords_per_frame, coords_average, center_image):
        # Create the train set
        train_set = []
        for i in range(len(coords_per_frame)):

            row = []
            coord_all = coords_per_frame[i]

            minimums = create_dataset.get_minimums_values(coord_all, 5)

            minimums_average = np.average(minimums)

            calculated_ang = create_dataset.calculate_ang(minimums_average, center_image, 910.0)
            
            row.append(calculated_ang)
            row = np.array(row).reshape(1, create_dataset.n_of_feature_per_row)
            train_set.append(row)

        return train_set

    """
    Methods for cleaning dataset
    """

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


    def calculate_ang(center_direction, center_image, focal_length):

        distance = abs(center_image - center_direction)
        ang = math.atan(distance / focal_length)
        return ang

