# GNN-PMB Tracker

This is the repo for GNN-PMB Tracker. The result of GNN-PMB + centerpoint detection from nuScenes validation data could be seen in gnnpmb_centerpoint_validation_result.out.

* See more details from our paper: Jianan Liu*, Liping Bai*, Yuxuan Xia, Tao Huang, Bing Zhu, Qing-Long Han, "GNN-PMB: A Simple but Effective Online 3D Multi-Object Tracker without Bells and Whistles" (https://arxiv.org/abs/2206.10255)


If you find our paper or code useful for you, please consider cite us by:
```
@article{liu2022gnn,
  title={GNN-PMB: A Simple but Effective Online 3D Multi-Object Tracker without Bells and Whistles},
  author={Liu, Jianan and Bai, Liping and Xia, Yuxuan and Huang, Tao and Zhu, Bing},
  journal={arXiv e-prints},
  pages={arXiv--2206},
  year={2022}
}
```

Code will be released soon. In case you are interested in getting the beta code before it is released, please contact jianan.liu@vitalent.se





## Environment
In the terminal, input the following command to install the environment: 
```
$ pip install -r requirements.txt
```

## Compile murty
Follow the instructions as follows:

1. Install eigen3.3. 
Eigen3 could be installed by writing:
```
$ sudo apt install libeigen3-dev
```

2. Clone the code of murty and store it into murty folder under current folder by writing:
```
$ git clone --recursive https://github.com/erikbohnsack/murty.git
```

3. Go to murty folder
```
$ cd murty
```

Then camke under murty folder by writing:
```
$ cmake .
```

4. compile murty according to cmakefile, by using make in the murty folder:
```
$ make
```

Note: step 3 and 4 can be alternatively replaced by "running setup.py under murty folder to install the murty lib", by writing:
```
$ pip install -e ./
```

If there is any more problem, the instruction is also provided in this repository https://github.com/erikbohnsack/murty



## Detection file
download the file and then put it in a folder of your choosing, make sure to specify the path to the folder.

### CenterPoint: 
Detections from CenterPoint can be downed from here(This is the detection file provided by the author of centerpoint. If there are better detection files, you can use those instead):
```
https://drive.google.com/drive/folders/14TvJegUV1BPCw5oWqEwR0AvXM8tjLsQt
```

### Megvii and PointPillars: 
Use the detection file provided by the organizer
```
megvii: https://www.nuscenes.org/data/detection-megvii.zip
pointpillars: https://www.nuscenes.org/data/detection-pointpillars.zip
```
Unzip the files



## Folders

### Detection file folder
Create a folder to put all the detection detection file. Name your files as the following
```
centerpoint_val.json
centerpoint_test.json
megvii_val.json
megvii_test.json
pointpillars_val.json
pointpillars_test.json
```
Make the folder `YOUR_DETECTION_FILE_FOLDER`, then put all the detection json file in that folder.

### Dataset folder for meta data
Create a folder `YOUR_DATASET_FOLDER`, download meta data of nuscenes test and trainval dataset. Unzip the these meta data into `YOUR_DATASET_FOLDER`, name them as the following folders:
```
v1.0-trainval
v1.0-test
```

### Result folder
Create a folder `YOUR_RESULT_FOLDER` to store the all the files related to tracking output.



## Run Tracker at Local Machine

### Preprocessing

To run preprocessing prodecure, write:
```
$ python3 utils/preprocessing.py --programme_file=YOUR_PROGRAMME_FOLDER --dataset_folder=YOUR_DATASET_FOLDER
```

### Run trackers
For instance, you want to evaluate the gnn-pmb tracker with trainval dataset, centerpoint detection. Here are the steps you have to take to do so.

```
$ python3 run_gnn_pmb_tracker.py --detection_file=YOUR_DETECTION_FILE_FOLDER --data_version='v1.0-trainval' --programme_file=YOUR_PROGRAMME_FOLDER --result_file=YOUR_RESULT_FOLDER --dataset_file=YOUR_DATASET_FOLDER
```

The parallel process is the number of threads you want to run, in our case we can run up to 36 threads but that might not be the capacity for other machines. Adjust accordingly e.g.
```
$ python3 run_gnn_pmb_tracker.py --parallel_process=36
```

If you want to visualize the result, specify the render_classes, in your local machine, e.g.
```
$ python3 run_gnn_pmb_tracker.py --render_classes='pedestrian'
```
