"""Verbatim copy of the evaluation logic of Event-AHU/DialBench `benchmark/benchmark.py`
(fetched 2026-10-07 via the GitHub contents API; repo license file: MIT, README says BSD 3-Clause).
Only the I/O, model, tqdm and image-saving lines are removed; every statement that computes a number is unchanged.
Used as the oracle in tests/test_dialbench_metrics.py (our port must match it exactly)."""
import re
import numpy as np


def extract_last_number(text):
    """
    从字符串中提取最后一个数字（整数或浮点数）。
    """
    matches = re.findall(r'[-+]?\d*\.\d+|[-+]?\d+', text)
    if matches:
        return float(matches[-1])
    else:
        return 0.0


def calculate_metrics(absolute_errors, ranges,
                      valid_theta_samples,
                      correct_count_theta, nonzero_gt_count_theta,
                      accuracy_epsilon_correct_count, processed_samples_count):
    metrics = {}

    if len(valid_theta_samples) > 0:
        theta_ratios = np.array(valid_theta_samples)
        theta_ratios = np.clip(theta_ratios, None, 100)
        metrics["theta"] = np.mean(theta_ratios)
    else:
        metrics["theta"] = None

    if len(absolute_errors) > 0 and len(ranges) > 0:
        gamma_ratios = np.array(absolute_errors) / np.array(ranges)
        gamma_ratios = np.clip(gamma_ratios, None, 10)
        metrics["gamma"] = np.mean(gamma_ratios)
    else:
        metrics["gamma"] = None

    if nonzero_gt_count_theta > 0:
        metrics["accuracy_theta"] = correct_count_theta / nonzero_gt_count_theta
    else:
        metrics["accuracy_theta"] = None

    if processed_samples_count > 0:
        metrics["accuracy_epsilon"] = accuracy_epsilon_correct_count / processed_samples_count
    else:
        metrics["accuracy_epsilon"] = None

    return metrics


def official_eval(pred_texts, readings, ranges_raw):
    """Re-statement of the inner loop of `main()` plus `load_data`'s ground-truth construction."""
    ground_truths = [extract_last_number(str(r)) for r in readings]   # load_data
    ranges = [float(r) for r in ranges_raw]                            # load_data
    absolute_errors, processed_ranges, valid_theta_samples = [], [], []
    correct_count_theta, nonzero_gt_count_theta, accuracy_epsilon_correct_count, processed_samples_count = 0, 0, 0, 0
    for original_idx, pred_text in enumerate(pred_texts):
        ground_truth = ground_truths[original_idx]
        range_value = ranges[original_idx]

        pred_number = extract_last_number(pred_text)
        error = abs(pred_number - ground_truth)

        absolute_errors.append(error)
        processed_ranges.append(range_value)

        epsilon = 0.01 * range_value
        if error <= epsilon:
            accuracy_epsilon_correct_count += 1

        if ground_truth != 0:
            nonzero_gt_count_theta += 1
            theta_sample = error / ground_truth
            valid_theta_samples.append(theta_sample)

            if theta_sample < 0.05:
                correct_count_theta += 1

        processed_samples_count += 1
    return calculate_metrics(absolute_errors, processed_ranges, valid_theta_samples,
                             correct_count_theta, nonzero_gt_count_theta,
                             accuracy_epsilon_correct_count, processed_samples_count)
