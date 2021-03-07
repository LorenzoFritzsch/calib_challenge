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

    """
    old_gray_gauss_filter = cv2.subtract(old_gray, old_gray_blurred)
    gray_gauss_filter = cv2.subtract(gray, gray_blurred)
    return old_gray_gauss_filter, gray_gauss_filter
    """
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

def clear_dataset(dataset, show_standard_deviation=False):
    dataset_standard_dev = get_standard_deviation(dataset)
    if show_standard_deviation:
        print("Standard deviation of dataset: " + str(dataset_standard_dev))
    for data in dataset:
        if data > dataset_standard_dev:
            dataset.remove(data)
    return dataset

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

        #print("Frame: " + str(round) + " Lost frames: " + str(lost_frames) + "\033[93m Frames with errors: " + str(frames_with_errors) + "\033[0m")

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

            local_coords_a_positive = []
            local_coords_b_positive = []
            local_coords_c_positive = []
            local_coords_d_positive = []

            local_coords_a_negative = []
            local_coords_b_negative = []
            local_coords_c_negative = []
            local_coords_d_negative = []

            for i, (new, old) in enumerate(zip(good_new, good_old)):

                a, b = new.ravel()
                c, d = old.ravel()


                m = 0
                if a != c:
                    m = (d - b)/(c - a)

                else:
                    continue

                """
                m_angulus = math.atan(m) * 180 / math.pi
                if abs(m_angulus) >= 70 or abs(m_angulus) <= 5:
                    continue
                """

                if b > car_area[0][1] or d > car_area[1][1]:
                    continue

                #print(abs(m_angulus))

                #Calculate actual distance of the point, if too far ignore it, probably wrong.

                # raw_angular_coefficients.append(m)
                # Better data management
                if m > 0:
                    local_coords_a_positive.append(int(a))
                    local_coords_b_positive.append(int(b))
                    local_coords_c_positive.append(int(c))
                    local_coords_d_positive.append(int(d))
                else:
                    local_coords_a_negative.append(int(a))
                    local_coords_b_negative.append(int(b))
                    local_coords_c_negative.append(int(c))
                    local_coords_d_negative.append(int(d))

                #TODO: WHAT IF I CALCULATE THE X,Y CENTER HERE, AND THEN I AVERAGE IT?, mi servono due rette minimo, qua ne ho una alla volta. Take prev? 

                #mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), green, 1)
                #frame = cv2.circle(frame, (int(a), int(b)), 5, green, -1)

            #Clear data before averaging it
            local_coords_a_positive = clear_dataset(local_coords_a_positive)
            local_coords_b_positive = clear_dataset(local_coords_b_positive)
            local_coords_c_positive = clear_dataset(local_coords_c_positive)
            local_coords_d_positive = clear_dataset(local_coords_d_positive)

            local_coords_a_negative = clear_dataset(local_coords_a_negative)
            local_coords_b_negative = clear_dataset(local_coords_b_negative)
            local_coords_c_negative = clear_dataset(local_coords_c_negative)
            local_coords_d_negative = clear_dataset(local_coords_d_negative)

            a_average_positive = np.average(local_coords_a_positive)
            b_average_positive = np.average(local_coords_b_positive)
            c_average_positive = np.average(local_coords_c_positive)
            d_average_positive = np.average(local_coords_d_positive)


            a_average_negative = np.average(local_coords_a_negative)
            b_average_negative = np.average(local_coords_b_negative)
            c_average_negative = np.average(local_coords_c_negative)
            d_average_negative = np.average(local_coords_d_negative)

            m_average_positive = (d_average_positive - b_average_positive)/(c_average_positive - a_average_positive)

            m_average_negative = (d_average_negative - b_average_negative) / (c_average_negative - a_average_negative)



            q_one = m_average_positive * c_average_positive - d_average_positive
            q_two = m_average_negative * c_average_negative - d_average_negative

            matrix_determinant = [[m_average_positive, 1], [m_average_negative, 1]]
            matrix_x = [[q_one, 1], [q_two, 1]]
            matrix_y = [[m_average_positive, q_one], [m_average_negative, q_two]]

            #print(matrix_determinant, matrix_x, matrix_y)

            determinant = np.linalg.det(matrix_determinant)
            x_det = np.linalg.det(matrix_x)
            y_det = np.linalg.det(matrix_y)

            x_center = abs(x_det / determinant)
            y_center = abs(y_det / determinant)

            # Use prev calculated data to make better guesses.
            x_in_bound = x_center < width
            y_in_bound = y_center < height
            coords_in_bounds = x_in_bound and y_in_bound

            #print(determinant, x_det, y_det, x_center, y_center)


            x_center_all_average = np.average(x_center_all)
            y_center_all_average = np.average(y_center_all)


            if not math.isnan(x_center) and coords_in_bounds:

                if math.isnan(x_center_all_average):
                    x_center_all_average = x_center
                    y_center_all_average = y_center
                else:
                    if abs(math.atan(m_average_positive) * 180 / math.pi) < 1 or abs(math.atan(m_average_negative) * 180 / math.pi) < 1:
                        #print("UNDER 1 DEGREE m!")
                        x_center = x_center_all_average
                        y_center = y_center_all_average
                    else:
                        mask = cv2.line(mask, (int(a_average_positive), int(b_average_positive)), (int(c_average_positive), int(d_average_positive)), green, 1)
                        mask = cv2.line(mask, (int(a_average_negative), int(b_average_negative)), (int(c_average_negative), int(d_average_negative)), green, 1)


                x_center_wa = (x_center + x_center_all_average) / 2
                y_center_wa = (y_center + y_center_all_average) / 2


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
        x_center_all = clear_dataset(x_center_all, True)
        y_center_all = clear_dataset(y_center_all, True)
        generation += 1
