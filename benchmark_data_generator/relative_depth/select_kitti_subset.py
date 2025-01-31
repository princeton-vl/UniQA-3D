import json
import numpy as np

# np.random.seed(1234)
# subset_size = 1000

np.random.seed(1235)
subset_size = 500

# find the json here: https://github.com/youmi-zym/CompletionFormer/blob/main/data_json/kitti_dc.json
split_json = ''

with open(split_json) as json_file:
    json_data = json.load(json_file)
    sample_list = json_data['train']

subset_ids = np.random.choice(np.arange(len(sample_list)), size=subset_size, replace=False)
subset_samples = [sample_list[id] for id in subset_ids]

with open('./kitti_%d.json' % subset_size, 'w') as f:
    json.dump(subset_samples, f)
