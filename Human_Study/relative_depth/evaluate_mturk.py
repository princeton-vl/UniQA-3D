import numpy as np
import pandas as pd
import utils
import os
import json

answer_mapping = {
    'Marker 1 (red) is farther': 1,
    'Marker 2 (green) is farther': 2
}

result_fns = ['Batch_500.csv', 'Batch_250_flipud.csv']
metadata_dirs = ['../../benchmark_data/relative_depth/kitti_1000_meta', '../../benchmark_data/relative_depth/kitti_500_flipud_meta']

valid_rad = 5.0
usr_acc_thresh = 0.5

for results_fn, meta_folder in zip(result_fns, metadata_dirs):
    csvFile = pd.read_csv(results_fn)[1:]

    image_fns = csvFile['Input.image_url'].values 
    answers_raw = csvFile['Answer.annotatedResult.keypoints'].values 
    usr_id = csvFile['WorkerId'].values 

    def convert_depth_click_raw(meta_folder, id, answer_raw, valid_radius=5.0):
        answer_raw = json.loads(answer_raw)
        valid = True

        meta_fn = os.path.join(meta_folder, id + '_meta.json')
        meta = utils.retrieve_metadata(meta_fn)
        loc = meta['sample_coord']
        loc1_h, loc1_w, loc2_h, loc2_w = int(loc[0][0]), int(loc[0][1]), int(loc[1][0]), int(loc[1][1])
        loc1 = np.array([loc1_h, loc1_w])
        loc2 = np.array([loc2_h, loc2_w])

        if len(answer_raw) != 1:
            return False, None, False

        loc_pred = np.array([answer_raw[0]["y"], answer_raw[0]["x"]])
        dist_to_marker_1 = np.linalg.norm(loc_pred - loc1)
        dist_to_marker_2 = np.linalg.norm(loc_pred - loc2)

        valid = dist_to_marker_1 < valid_radius or dist_to_marker_2 < valid_radius

        # print(loc1, loc2, loc_pred)
        depths = meta['depths']
        gt_relative_depth = 1 if float(depths[0]) > float(depths[1]) else 2

        user_ans = np.argmin([dist_to_marker_1, dist_to_marker_2]) + 1

        return valid, user_ans, ((gt_relative_depth == user_ans) and valid)

    image_ids = [f.split('_')[0] for f in image_fns]

    # there can be multiple answers (by different users) for a single image.
    image_ids_unique = np.unique(image_ids)
    answers_zipped = {id: [] for id in image_ids_unique}

    total_valid = 0
    total_raw = 0


    # rule out the users whose acc is too low
    usr_ids_unique = np.unique(usr_id)
    usr_stats = {uid: [] for uid in usr_ids_unique}
    usr_blacklist = []

    for id, a_raw, uid in zip(image_ids, answers_raw, usr_id):
        valid, a, is_correct = convert_depth_click_raw(meta_folder, id, a_raw, valid_radius=valid_rad)
        usr_stats[uid].append(is_correct)

    for uid in usr_ids_unique:
        if np.sum(usr_stats[uid]) / len(usr_stats[uid]) < usr_acc_thresh:
            usr_blacklist.append(uid)

    print(f'total users: {len(usr_ids_unique)}')
    print(f'total valid users: {len(usr_ids_unique) - len(usr_blacklist)}')

    for id, a_raw, uid in zip(image_ids, answers_raw, usr_id):
        valid, a, is_correct = convert_depth_click_raw(meta_folder, id, a_raw, valid_radius=valid_rad)
        # print(valid, a)
        # if valid:
        if valid and uid not in usr_blacklist:
            answers_zipped[id].append(a)

        total_raw += 1
        total_valid += int(valid)

    print('answer valid rate with %.1f px range: ' % valid_rad, total_valid / total_raw)

    result_dict = utils.cauculate_acc(meta_folder, answers_zipped)
    valid_ans_ids = result_dict['valid_ans_ids']
    csv_to_save = {'image_url': [id + '_rgb.png' for id in valid_ans_ids]}
    df = pd.DataFrame(csv_to_save)
    df.to_csv(results_fn.split('.')[0] + '_validclick.csv', index=False)

    dataset_name = results_fn.split('.')[0]
    np.savez(f'./results_{dataset_name}.npz', result_dict)
