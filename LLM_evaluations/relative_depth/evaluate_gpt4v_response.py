import numpy as np
import pandas as pd
import utils
import tqdm
import os
import json

# two parts: normal, and upside-down (geometric pertubation)
output_folders = ['./gpt_results', './gpt_results_flipud']
metadata_dirs = ['../../benchmark_data/relative_depth/kitti_1000_meta', '../../benchmark_data/relative_depth/kitti_500_flipud_meta']
image_lists = ['../../benchmark_data/relative_depth/kitti_1000_mturk_valid.csv', '../../benchmark_data/relative_depth/kitti_500_flipud_mturk_valid.csv']

for output_folder, meta_folder, image_list in zip(output_folders, metadata_dirs, image_lists):
    csvFile = pd.read_csv(image_list)
    image_fns = csvFile['image_url'].values
    image_ids = [f.split('_')[0] for f in image_fns]

    answers_zipped = {id: [] for id in image_ids}

    for image_name in image_fns:
        image_id = image_name.split('_')[0]
        result_fn = os.path.join(output_folder, image_id + '_gpt.json')

        with open(result_fn, 'r') as f:
            result = json.load(f)

        answer = int(result['short_answer'])
        answers_zipped[image_id].append(answer)

    utils.cauculate_guess_acc(meta_folder, answers_zipped)
    result_dict = utils.cauculate_acc(meta_folder, answers_zipped, show_curves=False)
    dataset_name = (image_list.split('/')[-1]).split('.')[0]
    np.savez(f'./results_gpt4v_{dataset_name}.npz', result_dict)