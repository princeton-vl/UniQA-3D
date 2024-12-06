import numpy as np
import pandas as pd
import os
import json
from scipy.stats import mode

answer_mapping = {
    'Marker 1 (red) is farther': 1,
    'Marker 2 (green) is farther': 2
}


print(os.getcwd())
results_fns = ['500_90a_1000ph.csv', 'flipud_250_90a_1000ph.csv']

usr_acc_thresh = 0.5

data_folders = ['../../benchmark_data/relative_camera_pose/DTU_500_mturk', '../../benchmark_data/relative_camera_pose/DTU_250_rot180_mturk']


def calculate_acc(data_folder, answers_zipped):
    valid_image_fns = []
    majority_answer_list = []
    ud_list = []
    raw_ans = {}
    correctness = []

    correct = 0
    total = 0

    correct_ud = 0
    total_ud = 0

    correct_lr = 0
    total_lr = 0

    for image_fn, answers in answers_zipped.items():
        if len(answers) == 0:
            continue

        folder_id = image_fn.split('/')[0]

        meta_fn = os.path.join(data_folder, folder_id, 'meta.json')
        with open(meta_fn, 'r') as f:
            meta = json.load(f)

        # gt answer
        gt_ans = 0 if meta['answer'].startswith('A') else 1
        answer_majority = mode(answers).mode

        cnt = mode(answers).count
        # user doesn't agree, ignore this result
        if cnt < len(answers):
            continue

        # we now know answers are valid, i.e., at least one answer and all workers agree
        valid_image_fns.append(image_fn) 
        raw_ans[id] = answer_majority
        majority_answer_list.append(answer_majority)

        if answer_majority == gt_ans:
            correct += 1
            correctness.append(1)
        else:
            correctness.append(0)
        total += 1

        if "up" in meta["answer"]:
            ud_list.append(1)
            total_ud += 1
            if answer_majority == gt_ans:
                correct_ud += 1
        elif "left" in meta["answer"]:
            ud_list.append(0)
            total_lr += 1
            if answer_majority == gt_ans:
                correct_lr += 1
        else:
            raise NotImplementedError


    total_acc = correct / total

    lr_acc = correct_lr / total_lr
    ud_acc = correct_ud / total_ud

    print('total number of images with valid click: ', total)
    print('total number of images with correct answer: ', correct)
    print('acc is', total_acc)
    print('acc for up/down pair is', ud_acc)
    print('acc for left/right pair is', lr_acc)

    return {
        'total_acc': total_acc,
        'valid_image_fns': valid_image_fns,
        'raw_ans': raw_ans,
        'ud_acc': ud_acc,
        'lr_acc': lr_acc,
        'correctness': correctness, 
        'majority_answer_list': majority_answer_list,
        'ud_list': ud_list,
    }

def inside_box(coord, box):
    if coord[0] >= box[0][0] and coord[0] <= box[1][0] and coord[1] >= box[0][1] and coord[1] <= box[1][1]:
        return True
    else:
        return False

def convert_depth_click_raw(data_folder, image_fn, answer_raw):
    multiple_clicks = False
    click_outside_box = False
    ans = None
    valid = False

    answer_raw = json.loads(answer_raw)

    folder_id = image_fn.split('/')[0]

    meta_fn = os.path.join(data_folder, folder_id, 'meta.json')
    with open(meta_fn, 'r') as f:
        meta = json.load(f)

    # gt answer
    gt_ans = 0 if meta['answer'].startswith('A') else 1

    box1_coord = meta['box_1']
    box2_coord = meta['box_2']

    if len(answer_raw) != 1:
        multiple_clicks = True

    else:
        loc_pred = np.array([answer_raw[0]["x"], answer_raw[0]["y"]])
        if inside_box(loc_pred, box1_coord):
            ans = 0
            valid = True
        elif inside_box(loc_pred, box2_coord):
            ans = 1
            valid = True
        else:
            click_outside_box = True
   
    return multiple_clicks, click_outside_box, valid, ans, gt_ans

for results_fn, data_folder in zip(results_fns, data_folders):
    csvFile = pd.read_csv(results_fn)[1:]

    image_fns = csvFile['Input.image_url'].values 
    answers_raw = csvFile['Answer.annotatedResult.keypoints'].values 
    usr_id = csvFile['WorkerId'].values 

    # there can be multiple answers (by different users) for a single image.
    image_fns_unique = np.unique(image_fns)
    answers_zipped = {id: [] for id in image_fns_unique}
    gt_answers = {id: None for id in image_fns_unique}

    total_valid = 0
    total_invalid_multiple_click = 0
    total_invalid_outside_box = 0
    total_raw = 0

    # rule out the users whose acc is too low
    usr_ids_unique = np.unique(usr_id)
    usr_stats = {uid: [] for uid in usr_ids_unique}
    usr_blacklist = []

    # create black list
    for image_fn, a_raw, uid in zip(image_fns, answers_raw, usr_id):
        multiple_clicks, click_outside_box, valid, ans, gt_ans = convert_depth_click_raw(data_folder, image_fn, a_raw)
        usr_stats[uid].append((ans == gt_ans))

    for uid in usr_id:
        if np.sum(usr_stats[uid]) / len(usr_stats[uid]) < usr_acc_thresh:
            usr_blacklist.append(uid)

    print('number of hits accepted by each user: ', sorted([len(stat) for stat in usr_stats.values()])[::-1])

    # calculate acc
    for image_fn, a_raw, uid in zip(image_fns, answers_raw, usr_id):
        multiple_clicks, click_outside_box, valid, ans, gt_ans = convert_depth_click_raw(data_folder, image_fn, a_raw)

        gt_answers[id] = gt_ans
        if valid and uid not in usr_blacklist:
            answers_zipped[image_fn].append(ans)

        total_raw += 1
        total_valid += int(valid)
        total_invalid_multiple_click += int(multiple_clicks)
        total_invalid_outside_box += int(click_outside_box)

    print('total answers: ', total_raw)
    print('total valid: ', total_valid)
    print('answers invalid because clicked more than 1 time: ', total_invalid_multiple_click)
    print('answers invalid because clicked outside either boxes: ', total_invalid_outside_box)

    result_dict = calculate_acc(data_folder, answers_zipped)
    valid_image_fns = result_dict['valid_image_fns']
    correctness = result_dict['correctness']
    majority_answer_list = result_dict['majority_answer_list']
    ud_list = result_dict['ud_list']


    csv_to_save = {
        'image_url': valid_image_fns,
        'Correctness': correctness,
        'Majority_Answer': majority_answer_list,
        'UD': ud_list,
    }

    df = pd.DataFrame(csv_to_save)
    df.to_csv(results_fn.split('.')[0] + '_validclick.csv', index=False)
