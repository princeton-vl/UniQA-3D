## Relative Depth
### Download KITTI
Download the following files and unzip under the `kitti_depth` folder:

[data_depth_annotated](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_annotated.zip), 
[data_depth_velodyne](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_velodyne.zip), 
[data_depth_selection](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_selection.zip)

Finally, download kitti raw images by:

```
cd datasets/kitti_depth
wget https://github.com/youmi-zym/CompletionFormer/files/12575038/kitti_archives_to_download.txt
wget -i kitti_archives_to_download.txt -P kitti_raw/
cd kitti_raw
unzip "*.zip"
```

The overall data directory is structured as follows:

```
kitti_depth
    ├──data_depth_annotated
    |     ├── train
    |     └── val
    ├── data_depth_velodyne
    |     ├── train
    |     └── val
    ├── data_depth_selection
    |     ├── test_depth_completion_anonymous
    |     |── test_depth_prediction_anonymous
    |     └── val_selection_cropped
    └── kitti_raw
          ├── 2011_09_26
          ├── 2011_09_28
          ├── 2011_09_29
          ├── 2011_09_30
          └── 2011_10_03
```
### Generate Images with Marks
Go to `benchmark_data_generator/relative_depth`. Run `render_kitti_image.py` or `render_kitti_flipud.py` (remember to replace the `PATH_TO_KITTI` variable 
to the directory you downloaded the KITTI dataset into).
