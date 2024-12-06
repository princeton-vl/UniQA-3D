import numpy as np
import json
from scipy.stats import mode
import os
import re

import seaborn as sn
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# mapping raw classes to simplified classes
CLASSES = ['road', 'building', 'vegetation', 'vehicle', 'traffic signs']
CLASSES_MAPPING = {
    0: 0,
    1: 0,
    2: 1,
    3: 1,
    4: 1,
    5: 4,
    6: 4,
    7: 4,
    8: 2,
    9: 2,
    10: 4,
    11: 4,
    12: 3,
    13: 3,
    14: 3,
    15: 3,
    16: 3,
    17: 3,
    18: 3
}


def read_pfm(path):
    """Read pfm file.

    Args:
        path (str): path to file

    Returns:
        tuple: (data, scale)
    """
    with open(path, "rb") as file:

        color = None
        width = None
        height = None
        scale = None
        endian = None

        header = file.readline().rstrip()
        if header.decode("ascii") == "PF":
            color = True
        elif header.decode("ascii") == "Pf":
            color = False
        else:
            raise Exception("Not a PFM file: " + path)

        dim_match = re.match(r"^(\d+)\s(\d+)\s$", file.readline().decode("ascii"))
        if dim_match:
            width, height = list(map(int, dim_match.groups()))
        else:
            raise Exception("Malformed PFM header.")

        scale = float(file.readline().decode("ascii").rstrip())
        if scale < 0:
            # little-endian
            endian = "<"
            scale = -scale
        else:
            # big-endian
            endian = ">"

        data = np.fromfile(file, endian + "f")
        shape = (height, width, 3) if color else (height, width)

        data = np.reshape(data, shape)
        data = np.flipud(data)

        return data, scale

def retrieve_metadata(meta_fn):
    with open(meta_fn, 'r') as f:
        meta = json.load(f)

    return meta

def get_answer_from_depth(meta_folder, image_id, depth_pred):
    meta_fn = os.path.join(meta_folder, image_id+'_meta.json')
    meta = retrieve_metadata(meta_fn)
    loc = meta['sample_coord']
    loc1_h, loc1_w, loc2_h, loc2_w = int(loc[0][0]), int(loc[0][1]), int(loc[1][0]), int(loc[1][1])
    depth_pred_1 = 1. / depth_pred[loc1_h, loc1_w]
    depth_pred_2 = 1. / depth_pred[loc2_h, loc2_w]

    pred_relative_depth = 1 if depth_pred_1 > depth_pred_2 else 2

    return pred_relative_depth


def cauculate_guess_acc(meta_folder, answers_zipped):
    valid_ans_ids = []

    correct_rand = 0
    total_rand = 0

    for id, answers in answers_zipped.items():
        if len(answers) == 0:
            continue

        meta_fn = os.path.join(meta_folder, id + '_meta.json')
        answer_majority = mode(answers).mode[0]

        cnt = mode(answers).count
        # user doesn't agree, ignore this result
        # if cnt <= len(answers) / 2:
        #    continue
        if cnt < len(answers):
            continue

        # we now know answers are valid, i.e., at least one answer and all workers agree
        valid_ans_ids.append(id) 

        meta = retrieve_metadata(meta_fn)
        depths = meta['depths']
        gt_relative_depth = 1 if float(depths[0]) > float(depths[1]) else 2
        
        # print(id)
        # print(gt_relative_depth)
        # print(answer_majority)

        loc = meta['sample_coord']
        loc1_h, loc1_w, loc2_h, loc2_w = int(loc[0][0]), int(loc[0][1]), int(loc[1][0]), int(loc[1][1])
        guess = 1 if loc1_h < loc2_h else 2

        if meta["sample_mode"] == "symmetric":
            pass
        elif meta["sample_mode"] == "random":
            total_rand += 1
            if guess == gt_relative_depth:
                correct_rand += 1
        else:
            raise NotImplementedError

    rand_acc = correct_rand / total_rand

    print('acc for guessing by y axis', rand_acc)


def cauculate_acc(meta_folder, answers_zipped, show_curves=False):
    valid_ans_ids = []
    raw_ans = {}

    correct = 0
    total = 0

    correct_symm = 0
    total_symm = 0

    correct_rand = 0
    total_rand = 0

    correct_semantics = np.zeros([len(CLASSES), len(CLASSES)])
    total_semantics = np.zeros_like(correct_semantics)

    correct_single_semantics = np.zeros(len(CLASSES))
    total_single_semantics = np.zeros_like(correct_single_semantics)

    depth_diff_min = 1.0
    depth_diff_max = 60.0
    depth_diff_bins = 5
    total_depth_bins = np.zeros(depth_diff_bins)
    correct_depth_bins = np.zeros(depth_diff_bins)
    depth_bin_centers = np.exp((np.arange(depth_diff_bins) + 0.5) / depth_diff_bins * (np.log(depth_diff_max) - np.log(depth_diff_min)) + np.log(depth_diff_min))

    for id, answers in answers_zipped.items():
        if len(answers) == 0:
            continue

        meta_fn = os.path.join(meta_folder, id + '_meta.json')
        answer_majority = mode(answers).mode[0]

        cnt = mode(answers).count
        # user doesn't agree, ignore this result
        # if cnt <= len(answers) / 2:
        #    continue
        if cnt < len(answers):
            continue

        # we now know answers are valid, i.e., at least one answer and all workers agree
        valid_ans_ids.append(id) 
        raw_ans[id] = answer_majority

        meta = retrieve_metadata(meta_fn)
        depths = meta['depths']
        gt_relative_depth = 1 if float(depths[0]) > float(depths[1]) else 2
        
        # print(id)
        # print(gt_relative_depth)
        # print(answer_majority)

        if answer_majority == gt_relative_depth:
            correct += 1
        total += 1

        classes_1, classes_2 = CLASSES_MAPPING[int(meta['classes_raw'][0])], CLASSES_MAPPING[int(meta['classes_raw'][1])]
        classes_1, classes_2 = sorted([classes_1, classes_2])

        # total_semantics[classes_1, classes_2] += 1
        total_semantics[classes_2, classes_1] += 1
        if answer_majority == gt_relative_depth:
            # correct_semantics[classes_1, classes_2] += 1
            correct_semantics[classes_2, classes_1] += 1

        total_single_semantics[classes_1] += 1
        total_single_semantics[classes_2] += 1
        if answer_majority == gt_relative_depth:
            correct_single_semantics[classes_1] += 1
            correct_single_semantics[classes_2] += 1

        depth_diff = np.abs(float(depths[0]) - float(depths[1]))
        bin_id = (np.log(depth_diff) - np.log(depth_diff_min)) / (np.log(depth_diff_max) - np.log(depth_diff_min))
        bin_id = np.clip(int(np.floor(depth_diff_bins * bin_id)), a_min=0, a_max=depth_diff_bins-1)
        total_depth_bins[bin_id] += 1
        if answer_majority == gt_relative_depth:
            correct_depth_bins[bin_id] += 1

        if meta["sample_mode"] == "symmetric":
            total_symm += 1
            if answer_majority == gt_relative_depth:
                correct_symm += 1
        elif meta["sample_mode"] == "random":
            total_rand += 1
            if answer_majority == gt_relative_depth:
                correct_rand += 1
        else:
            raise NotImplementedError


    total_acc = correct / total

    symm_acc = correct_symm / total_symm
    rand_acc = correct_rand / total_rand

    print('total number of images with valid click: ', total)
    print('acc is', total_acc)
    print('acc for symmetric pair is', symm_acc)
    print('acc for random pair is', rand_acc)

    depth_bins_acc = correct_depth_bins / total_depth_bins
    fig1, ax1 = plt.subplots()
    ax1.plot(depth_bin_centers, depth_bins_acc)
    ax1.set_xscale('log')
    ax1.set_xticks(depth_bin_centers)
    ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    if show_curves:
        plt.show()

    semantics_acc = correct_semantics / total_semantics
    semantics_acc_relative = semantics_acc / total_acc
    df_cm = pd.DataFrame(semantics_acc_relative, index = CLASSES,
                  columns = CLASSES)
    norm = plt.Normalize(0.8, 1.2)
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True, fmt='.2f', norm=norm)
    if show_curves:
        plt.show()

    return {
        'total_acc': total_acc,
        'valid_ans_ids': valid_ans_ids,
        'raw_ans': raw_ans,
        'depth_bins_info': {'depth_bins_acc': depth_bins_acc, 'depth_bin_centers': depth_bin_centers},
        'sampling_pattern_info': {'symm_acc': symm_acc, 'rand_acc': rand_acc},
        'semantics_acc': semantics_acc,
        'single_semantics_acc': correct_single_semantics / total_single_semantics
    }

    # return total_acc, symm_acc, rand_acc, total




