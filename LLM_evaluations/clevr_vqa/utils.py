import numpy as np
import json
from scipy.stats import mode
import os
import re

import seaborn as sn
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

import scipy.stats

import numpy as np

def create_bins(min_val, max_val, num_bins):
    # Generate evenly spaced bin edges
    bin_edges = np.linspace(min_val, max_val, num_bins + 1)
    return bin_edges

def find_bin_number(value, bin_edges):
    # Find which bin the value belongs to
    bin_number = np.digitize(value, bin_edges) - 1
    return bin_number

def compare_answers(correct, model_ans):
    
    model_ans = model_ans.lower().replace('.', '').replace(' ', '')
    model_ans = model_ans.replace('yes', 'true').replace('no', 'false')

    if isinstance(correct, bool):
        if correct: 
            correct = "true"
        else:
            correct = "false"
        
        if (correct == model_ans):
            return 1
        else:
            return 0

    elif isinstance(correct, int):
        if correct == 0:
            correct = "0"
        elif correct == 1:
            correct = "1"
        elif correct == 2:
            correct = "2"
        elif correct == 3:
            correct = "3"
        elif correct == 4:
            correct = "4"
        elif correct == 5:
            correct = "5"
        elif correct == 6:
            correct = "6"    

        if (correct == model_ans):
            return 1
        else:
            return 0

    else:
        if (correct == model_ans):
            return 1
        else:
            return 0

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h

def retrieve_metadata(meta_fn):
    with open(meta_fn, 'r') as f:
        meta = json.load(f)

    return meta

def cauculate_acc(meta_folder, answers_zipped, scene_json):
    valid_ans_ids = []
    raw_ans = {}

    is_correct = []

    scene_info = retrieve_metadata(scene_json)

    num_object_counts = {scene['image_filename'].split('.')[0]: len(scene['objects']) for scene in scene_info["scenes"]}

    max_num_objs = max(list(num_object_counts.values()))
    min_num_objs = min(list(num_object_counts.values()))

    min_q_length = 7
    max_q_length = 39
    num_bins = 6

    bin_edges = create_bins(min_q_length, max_q_length, num_bins)
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2.0

    is_correct_num_objects = [[] for _ in range(min_num_objs, max_num_objs+1)]
    is_correct_sentence_length = [[] for _ in range(num_bins)]

    raw_ans = []

    for id, answers in answers_zipped.items():
        if len(answers) == 0:
            continue

        meta_fn = os.path.join(meta_folder, id + '_gpt.json')
        # answer_majority = mode(answers).mode[0]

        # cnt = mode(answers).count
        # # user doesn't agree, ignore this result
        # # if cnt <= len(answers) / 2:
        # #    continue
        # if cnt < len(answers):
        #     continue

        answer_majority = mode(answers).mode[0]
        raw_ans.append(answer_majority)

        # we now know answers are valid, i.e., at least one answer and all workers agree
        valid_ans_ids.append(id) 
    
        meta = retrieve_metadata(meta_fn)
        gt_answer = meta['Correct Answer']

        question = meta['Asked Question']

        compare_results = [compare_answers(gt_answer, a) for a in answers]
       
        # if at least one of them answers it correctly:
        if 1 in compare_results:
            is_correct.append(1)
        else:
            is_correct.append(0)

        bin_id = find_bin_number(len(question.split(' ')), bin_edges)

        if 1 in compare_results:
            is_correct_sentence_length[bin_id].append(1)
        else:
            is_correct_sentence_length[bin_id].append(0)

        num_object = num_object_counts[id] - min_num_objs
        if 1 in compare_results:
            is_correct_num_objects[num_object].append(1)
        else:
            is_correct_num_objects[num_object].append(0)

    total_acc, total_95p = mean_confidence_interval(is_correct)

    sentence_length_acc, sentence_length_95p = \
        [mean_confidence_interval(data)[0] for data in is_correct_sentence_length], \
        [mean_confidence_interval(data)[1] for data in is_correct_sentence_length]

    num_objects_acc, num_objects_95p = \
        [mean_confidence_interval(data)[0] for data in is_correct_num_objects], \
        [mean_confidence_interval(data)[1] for data in is_correct_num_objects]

    print('total number of images with valid click: ', len(is_correct))
    print('acc is', total_acc)
    print('acc for sentenses lengths: ', bin_centers, sentence_length_acc)
    print('acc for num objects: ', list(range(min_num_objs, max_num_objs+1)), num_objects_acc)

    return {
        'is_correct': is_correct,
        'total_acc': total_acc,
        'total_95p': total_95p,
        'valid_ans_ids': valid_ans_ids,
        'raw_ans': raw_ans,
        'num_objects_stats': {'num_objects':list(range(min_num_objs, max_num_objs+1)), 'num_objects_acc': num_objects_acc, 'num_objects_95p': num_objects_95p},
        'sentence_length_stats': {'sentence_length':bin_centers, 'sentence_length_acc': sentence_length_acc, 'sentence_length_95p': sentence_length_95p},
    }