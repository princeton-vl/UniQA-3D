import numpy as np
import pandas as pd
import utils

results_fn = 'batch_500.csv'

meta_folder = '../../LLM_evaluations/clevr_vqa/gpt4_results'
scene_json  = '../../benchmark_data/clevr_vqa/CLEVR_scenes.json'

csvFile = pd.read_csv(results_fn)

image_fns = csvFile['Input.image_url'].values 
answers_raw = csvFile['Answer.category.label'].values 

answers_raw = [str(a).lower() for a in answers_raw] # convert int to str
answers_raw = [a.replace('yes', 'true').replace('no', 'false') for a in answers_raw]

image_ids = [f.split('.')[0] for f in image_fns]
answers = [a for a in answers_raw]

# there can be multiple answers (by different users) for a single image.
image_ids_unique = np.unique(image_ids)
answers_zipped = {id: [] for id in image_ids_unique}
for id, a in zip(image_ids, answers):
    answers_zipped[id].append(a)

result_dict = utils.cauculate_acc(meta_folder, answers_zipped, scene_json)
np.savez('./results_mturk.npz', result_dict)