from logging import raiseExceptions
import os
import json
from numpyencoder import NumpyEncoder
from utils.utils import nms, create_experiment_folder, initiate_submission_file, gen_measurement_of_this_class, initiate_classification_submission_file, readout_parameters
from trackers.PMBMGNN import PMBMGNN_Filter_Point_Target as pmbmgnn_tracker
from trackers.PMBMGNN import util as pmbmgnn_ulti
from datetime import datetime
from evaluate.util.utils import TrackingConfig, config_factory
from evaluate.evaluate_tracking_result import TrackingEval
import multiprocessing
import argparse
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_version', default='v1.0-trainval', help='choose dataset version between [v1.0-trainval][v1.0-test][v1.0-mini]')
    parser.add_argument('--detection_file',default='/home/Desktop/data/nuscenes/official_inference_result/centerpoint_val.json', help='directory for the inference file')
    parser.add_argument('--programme_file', default='/home/Desktop/Radar_Perception_Project/Project_5')
    parser.add_argument('--dataset_file', default='/home/Desktop/data/nuscenes')
    parser.add_argument('--parallel_process', default=5)
    parser.add_argument('--render_classes', default='')
    parser.add_argument('--result_file', default='/home/Desktop')
    parser.add_argument('--render_curves', default='False')
    parser.add_argument('--config_path',default='')
    parser.add_argument('--verbose',default='True')
    args = parser.parse_args()
    return args

def main(classification,token, out_file_directory_for_this_experiment):
    args=parse_args()
    dataset_info_file=args.programme_file+'/configs/dataset_info.json'
    config=args.programme_file+'/configs/gnnpmb_parameters.json'
    
    if args.data_version =='v1.0-trainval':
        set_info='val'
    elif args.data_version == 'v1.0-mini':
        set_info='mini_val'
    elif args.data_version == 'v1.0-test':
        set_info='test'
    else:
        raise KeyError('wrong data version')

    with open(dataset_info_file, 'rb') as f:
        dataset_info=json.load(f)

    orderedframe=dataset_info[set_info]['ordered_frame_info']

    timestamps=dataset_info[set_info]['time_stamp_info']

    egoposition=dataset_info[set_info]['ego_position_info']

    with open(args.detection_file, 'rb') as f:
        inference_meta= json.load(f)
    inference=inference_meta['results']

    with open(config, 'r') as f:
        parameters=json.load(f)

    birth_rate, P_s, P_d, use_ds_as_pd,clutter_rate, bernoulli_gating, extraction_thr, ber_thr, poi_thr, eB_thr, detection_score_thr, nms_score, confidence_score, P_init = readout_parameters(classification, parameters)
    if args.detection_file[21:]=='pointpillars_val.json':
        if classification=='car' or classification == 'pedestrian':
            detection_score_thr = 0.2
    
    filter_model = pmbmgnn_ulti.gen_filter_model(clutter_rate,P_s,P_d, classification, extraction_thr, ber_thr, poi_thr, eB_thr,bernoulli_gating, use_ds_as_pd, P_init)
    for scene_idx in range(len(list(orderedframe.keys()))):
        if scene_idx % args.parallel_process != token:
            continue
        scene_token = list(orderedframe.keys())[scene_idx]
        ordered_frames = orderedframe[scene_token]
        ego_info = egoposition[scene_token]
        
        gnnpmb_filter = pmbmgnn_tracker.PMBMGNN_Filter(filter_model)
        for frame_idx, frame_token in enumerate(ordered_frames):
            if frame_idx == 0:
                pre_timestamp = timestamps[scene_token][frame_idx]
            cur_timestamp = timestamps[scene_token][frame_idx]
            time_lag = (cur_timestamp - pre_timestamp)/1e6
            giou_gating = -0.5
            
            if frame_token in inference.keys():
                estimated_bboxes_at_current_frame = inference[frame_token]
            else:
                print('lacking inference file')
                break
            
            classification_submission = {}
            classification_submission['results']={}
            classification_submission['results'][frame_token] = []
            Z_k_all = gen_measurement_of_this_class(detection_score_thr, estimated_bboxes_at_current_frame, classification)
            
            result_indexes = nms(Z_k_all, threshold=nms_score)
            Z_k=[]
            for idx in result_indexes:
                Z_k.append(Z_k_all[idx])
            
            if frame_idx == 0:
                filter_predicted = gnnpmb_filter.predict_initial_step(Z_k, birth_rate)
            else:
                filter_predicted = gnnpmb_filter.predict(ego_info[str(frame_idx)],time_lag,filter_pruned, Z_k, birth_rate)
            
            filter_updated = gnnpmb_filter.update(Z_k, filter_predicted, confidence_score,giou_gating)
            
            if classification == 'pedestrian':
                if len(Z_k)==0:
                    estimatedStates_for_this_classification = gnnpmb_filter.extractStates_with_custom_thr(filter_updated, 0.7)
                else:
                    estimatedStates_for_this_classification = gnnpmb_filter.extractStates(filter_updated)
            else:
                estimatedStates_for_this_classification = gnnpmb_filter.extractStates(filter_updated)
            new =[]
            
            for idx in range(len(estimatedStates_for_this_classification['mean'])):
                instance_info = {}
                instance_info['sample_token'] = frame_token
                translation_of_this_target = [estimatedStates_for_this_classification['mean'][idx][0][0],
                                              estimatedStates_for_this_classification['mean'][idx][1][0], estimatedStates_for_this_classification['elevation'][idx]]
                instance_info['translation'] = translation_of_this_target
                instance_info['size'] = estimatedStates_for_this_classification['size'][idx]
                instance_info['rotation'] = estimatedStates_for_this_classification['rotation'][idx]
                instance_info['velocity'] = [estimatedStates_for_this_classification['mean']
                                             [idx][2][0], estimatedStates_for_this_classification['mean'][idx][3][0]]
                instance_info['tracking_id'] = estimatedStates_for_this_classification['classification'][idx]+'_'+str(
                    estimatedStates_for_this_classification['id'][idx])
                instance_info['tracking_name'] = estimatedStates_for_this_classification['classification'][idx]
                instance_info['tracking_score']=estimatedStates_for_this_classification['detection_score'][idx]
                
                new.append(instance_info)
            estimatedStates_for_this_classification = new
            classification_submission['results'][frame_token] = estimatedStates_for_this_classification

            filter_pruned = gnnpmb_filter.prune(filter_updated)

            pre_timestamp = cur_timestamp
            
            with open(out_file_directory_for_this_experiment+'/{}_{}.json'.format(frame_token, classification), 'w') as f:
                json.dump(classification_submission, f, cls=NumpyEncoder)
        print('done with {} scene {} process {}'.format(classification,scene_idx, token))

if __name__ == '__main__':
    arguments = parse_args()
    dataset_info_file=arguments.programme_file+'/configs/dataset_info.json'
    config=arguments.programme_file+'/configs/gnnpmb_parameters.json'
    if arguments.data_version =='v1.0-trainval':
        set_info='val'
    elif arguments.data_version == 'v1.0-mini':
        set_info='mini_val'
    elif arguments.data_version == 'v1.0-test':
        set_info='test'
    else:
        raise KeyError('wrong data version')
    classifications = ['bicycle','motorcycle',  'trailer', 'truck','bus','pedestrian','car']

    now = datetime.now()
    formatedtime = now.strftime("%Y-%m-%d-%H-%M-%S")
    out_file_directory_for_this_experiment = create_experiment_folder(arguments.result_file, formatedtime, set_info)
    for classification in classifications:
        inputarguments=[]
        for token in range(arguments.parallel_process):
            inputarguments.append((classification,token,out_file_directory_for_this_experiment))
        
        pool = multiprocessing.Pool(processes=arguments.parallel_process)
        pool.starmap(main,inputarguments)
        pool.close()
        print('{} is done'.format(classification))
    
    with open(dataset_info_file, 'r') as f:
        dataset_info=json.load(f)

    orderedframe=dataset_info[set_info]['ordered_frame_info']
    
    submission = initiate_submission_file(orderedframe)
    for classification in classifications:
        classification_submission = initiate_classification_submission_file(classification)
        for scene_idx in range(len(list(orderedframe.keys()))):
            scene_token = list(orderedframe.keys())[scene_idx]
            ordered_frames = orderedframe[scene_token]
            for frame_idx, frame_token in enumerate(ordered_frames):
                with open(out_file_directory_for_this_experiment+'/{}_{}.json'.format(frame_token, classification), 'r') as f:
                    results_all=json.load(f)
                classification_submission['results'][frame_token]=results_all['results'][frame_token]
            
        result_of_this_class = classification_submission['results']
        for frame_token in result_of_this_class:
            for bbox_info in result_of_this_class[frame_token]:
                submission['results'][frame_token].append(bbox_info)
    with open(out_file_directory_for_this_experiment+'/val_submission.json', 'w') as f:
        json.dump(submission, f, cls=NumpyEncoder)

    if arguments.data_version == 'v1.0-trainval' or arguments.data_version == 'v1.0-mini':
        
        result_path_ = os.path.expanduser(out_file_directory_for_this_experiment+'/val_submission.json')
        output_dir_ = os.path.expanduser(out_file_directory_for_this_experiment+'/nuscenes-metrics')
        eval_set_ = set_info
        dataroot_ = arguments.dataset_file
        version_ = arguments.data_version
        config_path = arguments.config_path
        render_curves_ = arguments.render_curves
        verbose_ = arguments.verbose
        render_classes_ = arguments.render_classes
    
        if config_path == '':
            cfg_ = config_factory(arguments.programme_file+'/configs/tracking_config.json')
        else:
            with open(config_path, 'r') as _f:
                cfg_ = TrackingConfig.deserialize(json.load(_f))
        nusc_eval = TrackingEval(config=cfg_, result_path=result_path_, eval_set=eval_set_, output_dir=output_dir_,
                                 nusc_version=version_, nusc_dataroot=dataroot_, verbose=verbose_,
                                 render_classes=render_classes_)
        nusc_eval.visualization_and_evaluation_of_tracking_results(render_curves=render_curves_)
