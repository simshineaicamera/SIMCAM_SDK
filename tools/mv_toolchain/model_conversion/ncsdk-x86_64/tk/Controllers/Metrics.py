# Copyright 2017 Intel Corporation.
# The source code, information and material ("Material") contained herein is
# owned by Intel Corporation or its suppliers or licensors, and title to such
# Material remains with Intel Corporation or its suppliers or licensors.
# The Material contains proprietary information of Intel or its suppliers and
# licensors. The Material is protected by worldwide copyright laws and treaty
# provisions.
# No part of the Material may be used, copied, reproduced, modified, published,
# uploaded, posted, transmitted, distributed or disclosed in any way without
# Intel's prior express written permission. No license under any patent,
# copyright or other intellectual property rights in the Material is granted to
# or conferred upon you, either expressly, by implication, inducement, estoppel
# or otherwise.
# Any license under such intellectual property rights must be express and
# approved by Intel in writing.

import numpy as np
import yaml
from csv import writer

import os
import sys
mdk_root = os.environ['HOME']

# Defines for Error report log
OKBLUE = '\033[94m'
OKGREEN = '\033[92m'
WARNING = '\033[93m'
FAIL = '\033[91m'
NORMAL = '\033[0m'
BOLD = '\033[1m'
PURPLE = '\033[95m'

NUM_OF_ATTR = 5
THRESHOLDS = [2, 1, 0, 1]


class CompareTestOutput:
    def __init__(self):
        line = OKBLUE + BOLD + 'TEST COMPARE: ' + NORMAL
        # print("----------------------------------------------------------------------------------")
        # print("{}:".format(line))
        # print("----------------------------------------------------------------------------------\n")
        self.NEW_REPORT = True
        self.test_index = 0

    # Debug_prints of data
    def debug_prints(self, matrix, fname):
        with open(fname + '.yaml', 'w') as temp:
            yaml.dump(matrix.tolist(), temp)

        with open(fname + '.yaml') as temp:
            loaded = yaml.load(temp)
        loaded = np.array(loaded)

    def metrics(self, a, b):
        ref = np.max(np.abs(b))
        total_values = int(len(b.flatten()))
        diff = np.abs(a - b)
        max_error = np.max(np.abs(a - b))
        mean_error = np.mean(np.abs(a - b))
        l2_error = np.sqrt(np.sum(np.square(a - b)) / total_values)

        if ref == 0:
            max_error = 0 if max_error == 0 else np.inf
            mean_error = 0 if mean_error == 0 else np.inf
            l2_error = 0 if l2_error == 0 else np.inf
        else:
            max_error = max_error / ref * 100
            mean_error = mean_error / ref * 100
            l2_error = l2_error / ref * 100
        percentage_wrong = len(
            np.extract(
                diff > 0.02 * ref,
                diff)) / total_values * 100
        sum_diff = np.sum(np.abs(a - b))
        return [max_error, mean_error, percentage_wrong, l2_error, sum_diff]

    def generate_report(self, report_file_obj, result, reference):
        obtained_val = [None] * NUM_OF_ATTR
        threshold_val = [None] * NUM_OF_ATTR
        attr_obtained = [None] * NUM_OF_ATTR
        attr_threshold = [None] * NUM_OF_ATTR

        out_array = [[], []]
        attr = [None] * NUM_OF_ATTR

        if(self.NEW_REPORT):
            attr[0] = 'min pixel accuracy'
            attr[1] = 'average pixel accuracy'
            attr[2] = 'percentage of correct values'
            attr[3] = 'pixel-wise l2 error'
            attr[4] = 'global sum difference'

            test_string = 'test index'
            out_array[0].append(test_string.upper())

            for attr_idx in range(0, len(attr)):
                attr_obtained[attr_idx] = 'M' + \
                    str(attr_idx + 1) + ' Obtained ' + attr[attr_idx]
                attr_obtained[attr_idx] = attr_obtained[attr_idx].upper()
                out_array[0].append(attr_obtained[attr_idx])

            for attr_idx in range(0, len(attr)):
                attr_threshold[attr_idx] = 'M' + \
                    str(attr_idx + 1) + ' threshold ' + attr[attr_idx]
                attr_threshold[attr_idx] = attr_threshold[attr_idx].upper()
                out_array[0].append(attr_threshold[attr_idx])

            string = 'Pass / Fail'
            out_array[0].append(string.upper())
            report_file_obj.writerow(out_array[0])
            self.NEW_REPORT = False
            if(result is None and reference is None):
                return (True)
        obtained_val = self.metrics(result.astype(np.float32), reference)

        test_status = self.matrix_comparison(obtained_val)

        threshold_val[0] = THRESHOLDS[0]
        threshold_val[1] = THRESHOLDS[1]
        threshold_val[2] = THRESHOLDS[2]
        threshold_val[3] = THRESHOLDS[3]
        threshold_val[4] = "Inf"

        if(self.NEW_REPORT == False):
            out_array[1].append(self.test_index)

            for attr_idx in range(0, len(attr)):
                out_array[1].append(obtained_val[attr_idx])

            for attr_idx in range(0, len(attr)):
                out_array[1].append(threshold_val[attr_idx])

            if(test_status):
                test_status_str = 'Pass'
            else:
                test_status_str = 'Fail'

            out_array[1].append(test_status_str)
            report_file_obj.writerow(out_array[1])

        return(test_status)

    def matrix_comparison(self, results):

        status = []
        for i in range(4):
            if results[i] > THRESHOLDS[i] or np.isnan(results[0]).any():
                status.append(FAIL + "Fail" + NORMAL)
            else:
                status.append("Pass")
        print("------------------------------------------------------------")
        print(" Obtained values ")
        print("------------------------------------------------------------")
        print(
            " Obtained Min Pixel Accuracy: {}% (max allowed={}%), {}".format(
                results[0],
                THRESHOLDS[0],
                status[0]))
        print(
            " Obtained Average Pixel Accuracy: {}% (max allowed={}%), {}".format(
                results[1],
                THRESHOLDS[1],
                status[1]))
        print(
            " Obtained Percentage of wrong values: {}% (max allowed={}%), {}".format(
                results[2],
                THRESHOLDS[2],
                status[2]))
        print(
            " Obtained Pixel-wise L2 error: {}% (max allowed={}%), {}".format(
                results[3],
                THRESHOLDS[3],
                status[3]))
        print(" Obtained Global Sum Difference: {}".format(results[4]))
        print("------------------------------------------------------------")
        return status[0] == 'Pass' and status[1] == 'Pass' and status[2] == 'Pass' and status[3] == 'Pass'


def compare_matricies(result, expected, filename=None, common_dimension=4):
    compare_obj = CompareTestOutput()
    f = open(filename, 'w+')
    csv = writer(f)
    compare_obj.generate_report(csv, result, expected)
    return


def check_match(output, expected):
    """
    Check our top1, top5 match for this image

    :param output:
    :param expected:
    :return:
    """
    data = output.flatten()
    sorted = np.argsort(data)
    top1 = True if (expected == sorted[0]) else False
    top5 = True if (
        expected in [
            sorted[0],
            sorted[1],
            sorted[2],
            sorted[3],
            sorted[4]]) else False

    return top1, top5

def compare_ssd_preds(expected_pred, resulted_pred, conf_tol, coord_tol):
    """
    Compare 2 predictions outputed by the DetectionLayer output of the SSD network.
    Prediction elements have the following meaning:
    [image_id | label | confidence | xmin | ymin | xmax | ymax]

    :param expected_pred: Expected prediction values.
    :param resulted_pred: Resulted prediciton values.
    :param conf_tol:  Tolerance for absolute diffrence between prediciton scores.
    :param coord_tol: Tolerance for absolute diffrence between prediction bounding
        box coordinates.
    :return: Match score computed as mean of absolute differences for the
        confidence, xmin, ymin, xmax and ymax fields. The score is np.inf if the
        image_id or the class lable do not match or if the score or coordinate
        absolute diffrence exceeds the specified tolerance. Lower is better.
    """

    e_img_id, e_label = expected_pred[0:2]
    r_img_id, r_label = resulted_pred[0:2]

    if(e_img_id != r_img_id or e_label != r_label):
        # Image ID or class label does not match.
        return np.inf

    match_abs_diffs = np.abs(expected_pred[2:] - resulted_pred[2:])
    match_score     = np.mean(match_abs_diffs)

    conf_abs_diff = match_abs_diffs[0]
    if(conf_abs_diff > conf_tol):
        return np.inf

    coords_abs_diff = match_abs_diffs[1:]
    if((coords_abs_diff > coord_tol).any()):
        return np.inf

    return match_score

def ssd_metrics(result, expected):
    """
    Compute metrics for DetectionOutput layer from the SSD Network.
    The output is a 2D matrix(table) of floating point values of size
    num_predicitons x 7. Each row is a prediction with the columns representing the
    following fields:
    [image_id | label | confidence | xmin | ymin | xmax | ymax]

    :param result:   2D np array of resulted predictions.
    :param expected: 2D np array of expected predictions.
    :return: None
    """

    print_pass = OKGREEN + BOLD + "Pass" + NORMAL
    print_fail = FAIL    + BOLD + "Fail" + NORMAL

    # Test if we can apply the metric.
    # The supported input shape is: [1, num_predictions, 7]
    if(len(result.shape) != 3 or len(expected.shape) != 3 
            or result.shape[0] != 1 or result.shape[2] != 7
            or expected.shape[0] != 1 or expected.shape[2] != 7):
        print(WARNING + "Invalid input for the SSD metric: " +
                "Please choose another metric." + NORMAL)
        return

    # Tolerances are hardcoded for the moment. They can be added as prameters if
    # required.
    num_preds_tolerance        = 0.02
    unmatched_tolerance        = 0.02
    out_of_order_tolerance     = 0.10
    multiple_matches_tolerance = 0.10
    conf_tolerance             = 0.01
    coord_tolerance            = 0.01

    # Remove 1st axis it should be one.
    result   = result[0]
    expected = expected[0]

    # Compare number of predictions.
    num_preds_expected = expected.shape[0]
    num_preds_resulted = result.shape[0]

    percent_diff_num_preds = ((num_preds_resulted - num_preds_expected) /
            num_preds_expected)
    pass_total_preds_test  = percent_diff_num_preds <= num_preds_tolerance

    # Try to match the resulted detections with the expected detections.
    match_scores = np.zeros((num_preds_resulted, num_preds_expected))
    for pred_i in range(0, num_preds_resulted):
        for expect_i in range(0, num_preds_expected):
            match_scores[pred_i, expect_i] = compare_ssd_preds(expected[expect_i, :],
                    result[pred_i, :], conf_tolerance, coord_tolerance)

    # Get the best matches.
    best_match_idx = np.argmin(match_scores, 1)

    matched = (np.zeros(num_preds_resulted) > 0)
    for pred_i, match_i in enumerate(best_match_idx):
        matched[pred_i] = np.isfinite(match_scores[pred_i, match_i])

    num_matched       = np.sum(matched)
    num_unmatched     = num_preds_resulted - num_matched
    unmatched_percent = ((num_unmatched / num_preds_resulted) \
            if num_preds_resulted != 0 else 1)
    pass_unmatched_test = unmatched_percent <= unmatched_tolerance

    num_out_of_order_matches = np.sum(np.logical_and(
            (np.arange(num_preds_resulted) - best_match_idx) != 0, matched)) // 2
    out_of_order_percent = (num_out_of_order_matches / num_matched) \
            if num_matched != 0 else 1
    pass_out_of_order_test = out_of_order_percent <= out_of_order_tolerance

    num_multiple_matches     = num_matched - len(np.unique(best_match_idx[matched]))
    multiple_matches_percent = (num_multiple_matches / num_matched) \
            if num_matched != 0 else 1
    pass_multiple_matches_test = multiple_matches_percent <= multiple_matches_tolerance

    # Display results.
    h0 = ("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
    hr = ("-------------------------------RESULT---------------------------------\n")
    he = ("------------------------------EXPECTED--------------------------------\n")
    h1 = ("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")

    table_header = ("(idx, match) | num | label |  score  |  "
                    + "_xmin_  _ymin_  _xmax_  _ymax_")
    print_format_str = ("({0:> 3d}, {1:> 3d}  ) | {2:> 3d} | {3:> 5d} | " +
        "{4:> 1.4f} | {5:> 1.4f} {6:> 1.4f} {7:> 1.4f} {8:> 1.4f}")

    # Set index = -1 for unmatched predictions.
    best_match_idx[np.logical_not(matched)] = -1
    best_match_idx_expected = np.argmin(match_scores, 0)
    for pred_i, match_i in enumerate(best_match_idx_expected):
        if(np.isinf(match_scores[match_i, pred_i])):
            best_match_idx_expected[pred_i] = -1

    print(h0 + hr + h1 + table_header)
    for pred_i in range(0, num_preds_resulted):
        color_output = print_format_str
        if(best_match_idx[pred_i] == -1):
            color_output = FAIL + print_format_str + NORMAL
        elif(best_match_idx[pred_i] != pred_i):
            # Matched out of order but at least matched.
            color_output = WARNING + print_format_str + NORMAL
        elif(best_match_idx[pred_i] == pred_i):
            # The condition is a bit superflous but just in case something really
            # wrong happend.
            color_output = OKGREEN + print_format_str + NORMAL

        print(color_output.format(pred_i, best_match_idx[pred_i],
            int(result[pred_i, 0]), int(result[pred_i, 1]), result[pred_i, 2],
            result[pred_i, 3], result[pred_i, 4],
            result[pred_i, 5], result[pred_i, 6]))

    print(h0 + he + h1 + table_header)
    for pred_i in range(0, num_preds_expected):
        color_output = print_format_str
        if(best_match_idx_expected[pred_i] == -1):
            color_output = FAIL + print_format_str + NORMAL
        elif(best_match_idx_expected[pred_i] != pred_i):
            # Matched out of order but at least matched.
            color_output = WARNING + print_format_str + NORMAL
        elif(best_match_idx_expected[pred_i] == pred_i):
            # The condition is a bit superflous but just in case something really
            # wrong happend.
            color_output = OKGREEN + print_format_str + NORMAL

        print(color_output.format(pred_i, best_match_idx_expected[pred_i],
            int(expected[pred_i, 0]), int(expected[pred_i, 1]), expected[pred_i, 2],
            expected[pred_i, 3], expected[pred_i, 4],
            expected[pred_i, 5], expected[pred_i, 6]))

    # Display statistics
    print("------------------------------------------------------------")
    print("Statistics:")
    print("------------------------------------------------------------")
    print(("Predictions total: result = {0:>3d}/{1:>3d} expected, " +
        "diff = {2:>+6.2%} (tolerance = {3:>6.2%}): " +
        (print_pass if pass_total_preds_test else print_fail))
        .format(num_preds_resulted, num_preds_expected, percent_diff_num_preds,
            num_preds_tolerance))
    print(("Predictions unmatched percentage:             " +
        "       {0:> 6.2%} (tolerance = {1:>6.2%}): " +
        (print_pass if pass_unmatched_test else print_fail))
        .format(unmatched_percent, unmatched_tolerance))

    print(("Predictions matched out of order percentage:  " +
        "       {0:> 6.2%} (tolerance = {1:>6.2%}): " +
        (print_pass if pass_out_of_order_test else print_fail))
        .format(out_of_order_percent, out_of_order_tolerance))

    print(("Predictions multiple matches percentage:      " +
        "       {0:> 6.2%} (tolerance = {1:>6.2%}): " +
        (print_pass if pass_multiple_matches_test else print_fail))
        .format(multiple_matches_percent, multiple_matches_tolerance))
    print("------------------------------------------------------------")

    pass_full_test = not (pass_total_preds_test and pass_unmatched_test and \
            pass_out_of_order_test and pass_multiple_matches_test)

    return pass_full_test
