"""
    CompletionFormer
    ======================================================================

    KITTI Depth Completion Dataset Helper
"""

"""
KITTI Depth Completion json file has a following format:

{
    "train": [
        {
            "rgb": "rawdata/2011_09_30/2011_09_30_drive_0018_sync/image_02/data/0000000188.png",
            "depth": "data_depth_velodyne/train/2011_09_30_drive_0018_sync/proj_depth/velodyne_raw/image_02/0000000188.png",
            "gt": "data_depth_annotated/train/2011_09_30_drive_0018_sync/proj_depth/groundtruth/image_02/0000000188.png",
            "K": "rawdata/2011_09_30/calib_cam_to_cam.txt"
        }, ...
    ],
    "val": [
        {
            "rgb": "data_depth_selection/val_selection_cropped/image/2011_10_03_drive_0047_sync_image_0000000761_image_02.png",
            "depth": "data_depth_selection/val_selection_cropped/velodyne_raw/2011_10_03_drive_0047_sync_velodyne_raw_0000000761_image_02.png",
            "gt": "data_depth_selection/val_selection_cropped/groundtruth_depth/2011_10_03_drive_0047_sync_groundtruth_depth_0000000761_image_02.png",
            "K": "data_depth_selection/val_selection_cropped/intrinsics/2011_10_03_drive_0047_sync_image_0000000761_image_02.txt"
        }, ...
    ],
    "test": [
        {
        }, ...
    ]
}

Reference : https://github.com/XinJCheng/CSPN/blob/master/nyu_dataset_loader.py
"""

PATH_TO_KITTI = None # replace this with the kitti_depth folder you downloaded
assert PATH_TO_KITTI is not None

import os
import numpy as np
import json
import random
import cv2
import tqdm

from PIL import Image
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset

CLASSES = {
    0: "road",
    1: "sidewalk",
    2: "building",
    3: "wall",
    4: "fence",
    5: "pole",
    6: "traffic light",
    7: "traffic sign",
    8: "vegetation",
    9: "terrain",
    10: "sky",
    11: "person",
    12: "rider",
    13: "car",
    14: "truck",
    15: "bus",
    16: "train",
    17: "motorcycle",
    18: "bicycle"
}

def read_depth(file_name):
    # loads depth map D from 16 bits png file as a numpy array,
    # refer to readme file in KITTI dataset
    assert os.path.exists(file_name), "file not found: {}".format(file_name)
    image_depth = np.array(Image.open(file_name))

    # Consider empty depth
    assert (np.max(image_depth) == 0) or (np.max(image_depth) > 255), \
        "np.max(depth_png)={}, path={}".format(np.max(image_depth), file_name)

    image_depth = image_depth.astype(np.float32) / 256.0
    return image_depth

# Reference : https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
def read_calib_file(filepath):
    """Read in a calibration file and parse into a dictionary."""
    data = {}

    with open(filepath, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    return data

class KITTI(Dataset):
    def __init__(self, split_file, symmetric_sample_ratio=0.5, margin=20,
                 thickness=1, mid_sym_margin=50, min_depth_diff=1.0, flipud=False):
        super(KITTI, self).__init__()
        self.split_json = split_file
        self.dir_data = PATH_TO_KITTI
        with open(self.split_json) as json_file:
            json_data = json.load(json_file)
            self.sample_list = json_data
            # self.sample_list = json_data['train']

        self.mode_bar = symmetric_sample_ratio * len(self.sample_list)
        self.margin = margin
        self.thickness = thickness
        self.mid_sym_margin = mid_sym_margin
        self.min_depth_diff = min_depth_diff
        # self.sample_mode = sample_mode
        self.flipud = flipud

    def __len__(self):
        return len(self.sample_list)

    def draw_marker(self, image, u, v, text='1', color=(0, 255, 255), thickness=1):
        radius = 20
        image = cv2.circle(image, (v, u), 20, color, thickness)
        image = cv2.line(image, (v-3*radius, u), (v+3*radius, u), color, thickness)
        image = cv2.line(image, (v, u-3*radius), (v, u+3*radius), color, thickness)
        image = cv2.putText(image, text, (v - int(1.5*radius), u - int(1.5*radius)), cv2.FONT_HERSHEY_SIMPLEX,
                            1, color, thickness, cv2.LINE_AA)

        return image

    def __getitem__(self, idx):
        rgb, semantics, _, gt, = self._load_data(idx)

        rgb = np.array(rgb)
        semantics = np.array(semantics)
        gt = np.array(gt)

        if self.flipud:
            rgb = np.ascontiguousarray(np.flipud(rgb))
            semantics = np.ascontiguousarray(np.flipud(semantics))
            gt = np.ascontiguousarray(np.flipud(gt))

        h, w, _ = rgb.shape

        gt_valid_mask = (gt > 0.0)

        # no samples in margins of the image
        gt_valid_mask[:self.margin] = 0.0
        gt_valid_mask[h-self.margin:] = 0.0
        gt_valid_mask[:, :self.margin] = 0.0
        gt_valid_mask[:, w-self.margin:] = 0.0

        gt_valid_mask_flipped = gt_valid_mask[:, ::-1]
        gt_valid_mask_both = gt_valid_mask * gt_valid_mask_flipped

        # we don't want to two markers to be too close, so leave the mid of the image blank.
        gt_valid_mask_both[:, w//2 - self.mid_sym_margin:w//2 + self.mid_sym_margin] = 0.0

        # we want the depth of the two markers to be at least self.min_depth_diff meters away
        gt_flipped = gt[:, ::-1]
        gt_diff_valid = (np.abs(gt - gt_flipped) > self.min_depth_diff)
        gt_valid_mask_both = gt_valid_mask_both * gt_diff_valid

        if idx < self.mode_bar:
            sample_mode = "symmetric"
        else:
            sample_mode = "random"

        if sample_mode == "random":
            (valid_ids_u, valid_ids_v) = gt_valid_mask.nonzero()
            sample_id = np.random.choice(np.array(len(valid_ids_u)), 2, replace=False)
            sample_u_1, sample_v_1 = valid_ids_u[sample_id[0]], valid_ids_v[sample_id[0]]
            sample_u_2, sample_v_2 = valid_ids_u[sample_id[1]], valid_ids_v[sample_id[1]]
            # print(sample_u_1, sample_v_1, gt[sample_u_1, sample_v_1])
            # print(sample_u_2, sample_v_2, gt[sample_u_2, sample_v_2])

        elif sample_mode == "symmetric":
            (valid_ids_u, valid_ids_v) = gt_valid_mask_both.nonzero()
            sample_id = np.random.randint(len(valid_ids_u))

            sample_u_1, sample_v_1 = valid_ids_u[sample_id], valid_ids_v[sample_id]
            sample_u_2, sample_v_2 = valid_ids_u[sample_id], w - valid_ids_v[sample_id] - 1
            # print(sample_u_1, sample_v_1, gt[sample_u_1, sample_v_1])
            # print(sample_u_2, sample_v_2, gt[sample_u_2, sample_v_2])

        else:
            raise NotImplementedError

        rgb_raw = np.copy(rgb)

        # rgb = self.draw_marker(rgb, sample_u_1, sample_v_1, '1', (0, 255, 255))
        rgb = self.draw_marker(rgb, sample_u_1, sample_v_1, '1', (255, 0, 0), thickness=self.thickness)
        rgb = self.draw_marker(rgb, sample_u_2, sample_v_2, '2', (0, 255, 0), thickness=self.thickness)

        depth_1, depth_2 = gt[sample_u_1, sample_v_1], gt[sample_u_2, sample_v_2]
        class_1_raw, class_2_raw = semantics[sample_u_1, sample_v_1], semantics[sample_u_2, sample_v_2]
        class_1, class_2 = CLASSES[class_1_raw], CLASSES[class_2_raw]

        assert depth_1 > 0.0 and depth_2 > 0.0

        output = {
                  'raw': rgb_raw,
                  'rgb': rgb,
                  'depths': (depth_1, depth_2),
                  'classes_raw': (str(class_1_raw), str(class_2_raw)),
                  'classes': (class_1, class_2),
                  'file_paths': self.sample_list[idx],
                  'sample_mode': sample_mode,
                  'sample_coord': ((str(sample_u_1), str(sample_v_1)), (str(sample_u_2), str(sample_v_2)))
                  }

        return output

    def _load_data(self, idx):
        path_rgb = os.path.join(self.dir_data,
                                self.sample_list[idx]['rgb'])
        path_depth = os.path.join(self.dir_data,
                                  self.sample_list[idx]['depth'])
        path_gt = os.path.join(self.dir_data,
                               self.sample_list[idx]['gt'])
        path_calib = os.path.join(self.dir_data,
                                  self.sample_list[idx]['K'])
        path_semantics = os.path.join(self.dir_data,
                                      (self.sample_list[idx]['rgb']).replace('kitti_raw', 'semantic_maps'))

        depth = read_depth(path_depth)
        gt = read_depth(path_gt)

        # if self.mode in ['train', 'val']:
        #     calib = read_calib_file(path_calib)
        #     if 'image_02' in path_rgb:
        #         K_cam = np.reshape(calib['P_rect_02'], (3, 4))
        #     elif 'image_03' in path_rgb:
        #         K_cam = np.reshape(calib['P_rect_03'], (3, 4))
        #     K = [K_cam[0, 0], K_cam[1, 1], K_cam[0, 2], K_cam[1, 2]]
        # else:
        #     f_calib = open(path_calib, 'r')
        #     K_cam = f_calib.readline().split(' ')
        #     f_calib.close()
        #     K = [float(K_cam[0]), float(K_cam[4]), float(K_cam[2]),
        #          float(K_cam[5])]

        rgb = Image.open(path_rgb)
        semantics = Image.open(path_semantics)
        depth = Image.fromarray(depth.astype('float32'), mode='F')
        gt = Image.fromarray(gt.astype('float32'), mode='F')

        w1, h1 = rgb.size
        w2, h2 = depth.size
        w3, h3 = gt.size

        assert w1 == w2 and w1 == w3 and h1 == h2 and h1 == h3

        # return rgb, semantics, depth, gt, K
        return rgb, semantics, depth, gt

def save_metadata(path, sample):
    out_dict = {
        'file_paths': sample['file_paths'],
        'depths': ('%.3f' % sample['depths'][0], '%.3f' % sample['depths'][1]),
        'classes': sample['classes'],
        'classes_raw': sample['classes_raw'],
        'sample_mode': sample['sample_mode'],
        'sample_coord': sample['sample_coord'],
    }
    with open(path, 'w') as f:
        json.dump(out_dict, f)

if __name__ == "__main__":
    seed = 1234
    subset_size = 500
    base_path_rgb = 'kitti_%d_flipud_rgb' % subset_size
    base_path_raw = 'kitti_%d_flipud_raw' % subset_size
    base_path_meta = 'kitti_%d_flipud_meta' % subset_size

    os.makedirs(base_path_rgb, exist_ok=True)
    os.makedirs(base_path_raw, exist_ok=True)
    os.makedirs(base_path_meta, exist_ok=True)

    np.random.seed(seed)
    torch.manual_seed(seed)

    tgt_dataset = KITTI('kitti_%d.json' % subset_size, symmetric_sample_ratio=0.5, thickness=2, flipud=True)
    print('dataset length: ', len(tgt_dataset))

    for idx in tqdm.tqdm(range(len(tgt_dataset))):
        sample = tgt_dataset.__getitem__(idx)

        rgb_path = os.path.join(base_path_rgb, str(idx).zfill(6) + '_rgb.png')
        raw_path = os.path.join(base_path_raw, str(idx).zfill(6) + '_raw.png')
        meta_path = os.path.join(base_path_meta, str(idx).zfill(6) + '_meta.json')

        rgb = sample['rgb']
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(rgb_path, bgr)

        raw = sample['raw']
        bgr = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
        cv2.imwrite(raw_path, bgr)

        save_metadata(meta_path, sample)

        # print(sample['depths'], sample['classes'])

