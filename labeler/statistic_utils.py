import numpy as np
import math
import scipy.stats

class statistic_utils:


    def get_only_statistically_viable_coords(dataset_x, dataset_y):
        average_x = np.average(dataset_x)
        average_y = np.average(dataset_y)

        stand_dev_x = statistic_utils.get_standard_deviation(dataset_x)
        stand_dev_y = statistic_utils.get_standard_deviation(dataset_y)

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


    def remove_val_outside_standard_dev(dataset_x, dataset_y):
        # Backward Elimination with standard deviation as significance level.

        max_val_x, min_val_x = statistic_utils.get_max_min_value_considering_standard_dev(dataset_x)
        max_val_y, min_val_y = statistic_utils.get_max_min_value_considering_standard_dev(dataset_y)

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

    def get_standard_deviation(dataset):
        mean = np.average(dataset)
        n = len(dataset)
        return math.sqrt(sum((x - mean) ** 2 for x in dataset) / n)


    def get_max_min_value_considering_standard_dev(dataset):

        average = np.average(dataset)

        standard_dev = statistic_utils.get_standard_deviation(dataset)

        max_val = average + standard_dev
        min_val = average - standard_dev

        return max_val, min_val

