import base64
import requests
import os

import numpy as np
import pandas as pd
import json
import tqdm

# replace this with your own OpenAI API key!
API_KEY=None
assert API_KEY is not None

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def get_response(image_path):
    # Getting the base64 string
    base64_image = encode_image(image_path)

    headers = {
      "Content-Type": "application/json",
      "Authorization": f"Bearer {API_KEY}"
    }

    payload = {
      "model": "gpt-4-vision-preview",
      "messages": [
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": "There are two markers in the image. Which point is farther away from the camera? First answer either 1 or 2, with no additional content, and then explain your answer in separate sentences."
            },
            {
              "type": "image_url",
              "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
              }
            }
          ]
        }
      ],
      "max_tokens": 300
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    return response

# two parts: normal, and upside-down (geometric pertubation)
output_folders = ['./gpt_results', './gpt_results_flipud']
image_dirs = ['../../benchmark_data/relative_depth/kitti_1000_rgb', '../../benchmark_data/relative_depth/kitti_500_flipud_rgb']
image_lists = ['../../benchmark_data/relative_depth/kitti_1000_filelist.csv', '../../benchmark_data/relative_depth/kitti_500_flipud_filelist.csv']

for output_folder, image_dir, image_list in zip(output_folders, image_dirs, image_lists):
    csvFile = pd.read_csv(image_list)
    image_fns = csvFile['image_url'].values

    for image_name in tqdm.tqdm(image_fns):

        result_fn = os.path.join(output_folder, image_name.split('_')[0] + '_gpt.json')

        # Path to your image
        if os.path.exists(result_fn):
            print('skip because already processed: ', image_name)
            continue

        response = get_response(os.path.join(image_dir, image_name))
        # print(response.json())

        answer = response.json()["choices"][0]["message"]['content']
        result = {'img': image_name, 'full_answer': answer, 'short_answer': answer[0]}

        with open(result_fn, 'w') as f:
            json.dump(result, f)