import os
import json
import utils
import numpy as np

# Specify the directory containing JSON files
directory_paths = ["./gpt4_results"]
# directory_paths = ["./gpt4o_results", "./gpt4_results", "./gemini_results", "./mdetr_results"]
scene_json  = '../../benchmark_data/clevr_vqa/CLEVR_scenes.json'

for directory_path in directory_paths:
    # List all files in the directory
    files = os.listdir(directory_path)

    # Filter JSON files
    json_files = sorted([file for file in files if file.endswith('.json')])

    image_ids = [f'CLEVR_new_{idx:06}' for idx in range(len(json_files))]
    answers_zipped = {id: [] for id in image_ids}

    count = 0
    # Read and print each JSON file
    for idx, json_file in enumerate(json_files):
        file_path = os.path.join(directory_path, json_file)
        with open(file_path, 'r') as f:
            data = json.load(f)
            if "gemini" in directory_path:
                gpt =  data['gemini_short_answer'].lower()
            else:
                gpt = data['GPT Answer'].split('\n')[0].lower()

            gpt = gpt.replace('.', '')
            gpt = gpt.replace(' ', '')
            gpt = gpt.replace('yes', 'true')
            gpt = gpt.replace('no', 'false')

        image_id = f'CLEVR_new_{idx:06}'
        answers_zipped[image_id].append(gpt)

    result_dict = utils.cauculate_acc("./gpt4_results", answers_zipped, scene_json)
    np.savez(f'./results_{directory_path.split("/")[-1]}.npz', result_dict)

