import os, numpy as np, nuscenes, argparse, json
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits
from copy import deepcopy
from tqdm import tqdm
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box

def instance_info2bbox_array(info):
    translation = info.center.tolist()
    size = info.wlh.tolist()
    rotation = info.orientation.q.tolist()
    return translation + size + rotation

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--programme_file', default='/home/Desktop/Radar_Perception_Project/Project_5')
    parser.add_argument('--dataset_folder',default='/home/Desktop/mmdetection3d/data/nuscenes')
    args = parser.parse_args()
    return args

def main(args):
    dataset_versions=['v1.0-trainval','v1.0-test']
    for data_version in dataset_versions:
        nusc = NuScenes(version=data_version, dataroot=args.dataset_folder, verbose=True)
        scenes=nusc.scene
        if data_version =='v1.0-trainval':
            set_info='val'
        elif data_version == 'v1.0-mini':
            set_info='mini_val'
        else:
            set_info='test'
    
        scene_names = splits.create_splits_scenes()[set_info]
        pbar = tqdm(total=len(scene_names))
        
        if os.path.exists(args.programme_file+'/configs/dataset_info.json'):
            with open(args.programme_file+'/configs/dataset_info.json', 'rb') as f:
                aggregated_info = json.load(f)
        else:
            aggregated_info={}
    
        aggregated_info[set_info]={}
        
        aggregated_info[set_info]['ordered_frame_info']={}   
        
        aggregated_info[set_info]['time_stamp_info']={}
        
        aggregated_info[set_info]['ego_position_info']={}
        
        aggregated_info[set_info]['sensor_calibration_info']={}
        
        aggregated_info[set_info]['ground_truth_IDS']={}
        aggregated_info[set_info]['ground_truth_inst_types']={}
        aggregated_info[set_info]['ground_truth_bboxes']={}
    
        for scene_index, scene_info in enumerate(scenes):
            scene_name = scene_info['name']
            scene_token = scene_info['token']
            if scene_name not in scene_names:
                continue
            
            aggregated_info[set_info]['ordered_frame_info'][scene_token]=[]
            
            
            aggregated_info[set_info]['time_stamp_info'][scene_token]=[]
    
            
            aggregated_info[set_info]['ego_position_info'][scene_token]={}
            
            aggregated_info[set_info]['sensor_calibration_info'][scene_token]={}
            
            aggregated_info[set_info]['ground_truth_IDS'][scene_token]=[]
            aggregated_info[set_info]['ground_truth_inst_types'][scene_token]=[]
            aggregated_info[set_info]['ground_truth_bboxes'][scene_token]=[]
    
    
            first_sample_token = scene_info['first_sample_token']
            last_sample_token = scene_info['last_sample_token']
            frame_data = nusc.get('sample', first_sample_token)
            cur_sample_token = deepcopy(first_sample_token)
            
            frame_index = 0
            while True:
                
                frame_data = nusc.get('sample', cur_sample_token)
                aggregated_info[set_info]['ordered_frame_info'][scene_token].append(cur_sample_token)
                aggregated_info[set_info]['time_stamp_info'][scene_token].append(frame_data['timestamp'])
                lidar_token = frame_data['data']['LIDAR_TOP']
                frame_ids, frame_types, frame_bboxes = list(), list(), list()
                if data_version == 'v1.0-trainval' or data_version=='v1.0-mini':
                    instances = nusc.get_boxes(lidar_token)
            
                    for inst in instances:
                        frame_ids.append(inst.token)
                        frame_types.append(inst.name)
                        frame_bboxes.append(instance_info2bbox_array(inst))
                    aggregated_info[set_info]['ground_truth_IDS'][scene_token].append(frame_ids)
                    aggregated_info[set_info]['ground_truth_inst_types'][scene_token].append(frame_types)
                    aggregated_info[set_info]['ground_truth_bboxes'][scene_token].append(frame_bboxes)
    
                lidar_data = nusc.get('sample_data', lidar_token)
                calib_token = lidar_data['calibrated_sensor_token']
                calib_pose = nusc.get('calibrated_sensor', calib_token)
                aggregated_info[set_info]['sensor_calibration_info'][scene_token][str(frame_index)] = calib_pose['translation'] + calib_pose['rotation']
                ego_token = lidar_data['ego_pose_token']
                ego_pose = nusc.get('ego_pose', ego_token)
                aggregated_info[set_info]['ego_position_info'][scene_token][str(frame_index)] = ego_pose['translation'] + ego_pose['rotation']
                
                cur_sample_token = frame_data['next']
                if cur_sample_token == '':
                    break
                frame_index += 1
            pbar.update(1)
        pbar.close()
    
        with open(args.programme_file+'/configs/dataset_info.json', 'w') as f:
            json.dump(aggregated_info, f)
        f.close()

if __name__ == '__main__':
    args=parse_args()

    main(args)
