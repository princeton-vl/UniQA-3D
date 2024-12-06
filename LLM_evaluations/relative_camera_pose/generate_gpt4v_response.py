import base64
import requests
import os

import numpy as np
import pandas as pd
import json
import tqdm

import time

# replace this with your own OpenAI API key!
API_KEY=None
assert API_KEY is not None

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_response(data_path):
    # OpenAI API Key
    api_key = API_KEY

    # Getting the base64 string
    base64_image_1 = encode_image(os.path.join(data_path, 'Image1.png'))
    base64_image_2 = encode_image(os.path.join(data_path, 'Image2.png'))

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    with open(os.path.join(data_path, 'meta.json')) as f:
        meta = json.load(f)
        question = meta['question']
        gt_answer = meta['answer'][:-1].replace(',', ':') # discard the peroid

    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image_1}"
                        }
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image_2}"
                        }
                    },
                    {
                        "type": "text",
                        "text": question
                    }
                ]
            }
        ],
        "max_tokens": 300
    }
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    return response, gt_answer


output_folders = ["./gpt_responses", "./gpt_responses_rot180"]
image_dirs = ["./DTU_500_mturk", "./DTU_250_rot180_mturk"]
num_samples_ls = [500, 250]


for output_folder, image_dir, num_samples in zip(output_folders, image_dirs, num_samples_ls):
    total_correct = 0
    print(f"Processing {image_dir} with {num_samples} samples")
    for i in tqdm.tqdm(range(num_samples)):
        sample_dir = os.path.join(image_dir, str(i).zfill(4))

        result_fn = os.path.join(output_folder, str(i).zfill(4) +'_gpt.json')

        if os.path.exists(result_fn):
            print('skip because already processed: ', sample_dir)
            continue

        # response = get_response(image_name)
        response, gt_answer = get_response(sample_dir)
        print(response.json())
        print(gt_answer)
        answer = response.json()["choices"][0]["message"]['content']

        if gt_answer[0:2] in answer:
            correct = True
            total_correct += 1
        else:
            correct = False

        result = {'gpt_answer': answer, 'gt_answer': gt_answer, 'correct': correct}

        print(correct)

        with open(result_fn, 'w') as f:
            json.dump(result, f)

        # to avoid token mer min limit
        time.sleep(4)

    if image_dir == "./DTU_500_mturk":
        print('Relative Camera Pose Accuracy: ', total_correct / num_samples)
    else:
        print('Upside-down Relative Camera Pose Accuracy: ', total_correct / num_samples)
