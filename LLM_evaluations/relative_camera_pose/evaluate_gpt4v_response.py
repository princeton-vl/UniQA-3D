import base64
import requests
import os

import numpy as np
import pandas as pd
import json
import tqdm

import time


output_folders = ["./gpt_responses", "./gpt_responses_rot180"]
image_dirs = ["./DTU_500_mturk", "./DTU_250_rot180_mturk"]
num_samples_ls = [500, 250]


for output_folder, image_dir, num_samples in zip(output_folders, image_dirs, num_samples_ls):
    total_correct = 0
    for i in tqdm.tqdm(range(num_samples)):
        sample_dir = os.path.join(image_dir, str(i).zfill(4))

        result_fn = os.path.join(output_folder, str(i).zfill(4) +'_gpt.json')

        with open(result_fn, 'r') as f:
            result = json.load(f)

        if result['correct']:
            total_correct += 1

    if image_dir == "./DTU_500_mturk":
        print('Relative Camera Pose Accuracy: ', total_correct / num_samples)
    else:
        print('Upside-down Relative Camera Pose Accuracy: ', total_correct / num_samples)
