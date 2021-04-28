import sys

import cv2

import numpy as np

from statistic_utils import statistic_utils
from get_center_of_direction import get_center_of_direction 
from create_dataset import create_dataset
from params import params

class predict:

    def ann_predict(ann, max_video_number):

        for video_number in range(max_video_number):

            print("\nVideo: " + str(video_number))

            video_captured = cv2.VideoCapture(params.root_dir_labeled + str(video_number) + '.hevc')

            dict_frames = get_center_of_direction.get_center_of_direction_for_each_frame(video_captured, video_number)

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

                x_statistically_viable, y_statistically_viable = statistic_utils.get_only_statistically_viable_coords(x_saved, y_saved)
                x_in_standard_dev, y_in_standard_dev = statistic_utils.remove_val_outside_standard_dev(x_statistically_viable, y_statistically_viable)


                x_calc = np.average(x_in_standard_dev)
                y_calc = np.average(y_in_standard_dev)

                calculated_pitch = create_dataset.calculate_ang(y_calc, y_center_image, 910.0)
                calculated_yaw = create_dataset.calculate_ang(x_calc, x_center_image, 910.0)

                predicted_pitch = ann.predict(np.array([calculated_pitch]).reshape(1, 1))
                predicted_yaw = ann.predict(np.array([calculated_yaw]).reshape(1, 1))

                pitch_yaw_predicted.append([predicted_pitch, predicted_yaw])

            f = open("labeled/"+str(video_number)+".txt", "w")
            f.write(str(np.array(pitch_yaw_predicted).reshape(-1, 2)).replace("[", "").replace("]", ""))
            f.close()

            dict_frames = {}
