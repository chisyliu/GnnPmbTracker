import os
import copy
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
from cv2 import VideoWriter, VideoWriter_fourcc, imread
import numpy as np
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box
import torch
import os, numpy as np, json
import math
import numpy as np
from copy import deepcopy
from shapely.geometry import Polygon
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import numba
from scipy.spatial import ConvexHull
import numpy as np
from os import path
import glob
from tqdm import tqdm
import pickle
import torch
from copy import deepcopy
import json

def compute_box_3d(obj):
    R = roty(obj.yaw)    

    
    l = obj.l
    w = obj.w
    h = obj.h
    
    
    x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2]
    y_corners = [0,0,0,0,-h,-h,-h,-h]
    z_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2]
    
    
    corners_3d = np.dot(R, np.vstack([x_corners,y_corners,z_corners]))

    corners_3d[0,:] = corners_3d[0,:] + obj.X
    corners_3d[1,:] = corners_3d[1,:] + obj.Y
    corners_3d[2,:] = corners_3d[2,:] + obj.Z
    

    return np.transpose(corners_3d)

def box3doverlap(aa, bb, criterion='union'):
	aa_3d = compute_box_3d(aa)
	bb_3d = compute_box_3d(bb)

	iou3d, iou2d = box3d_iou(aa_3d, bb_3d, criterion=criterion)
	
	return iou3d

class BBox:
    def __init__(self, x=None, y=None, z=None, h=None, w=None, l=None, o=None):
        self.x = x      
        self.y = y      
        self.z = z      
        self.h = h      
        self.w = w      
        self.l = l      
        self.o = o      
        self.s = None   
    
    def __str__(self):
        return 'x: {}, y: {}, z: {}, heading: {}, length: {}, width: {}, height: {}, score: {}'.format(
            self.x, self.y, self.z, self.o, self.l, self.w, self.h, self.s)
    
    @classmethod
    def bbox2dict(cls, bbox):
        return {
            'center_x': bbox.x, 'center_y': bbox.y, 'center_z': bbox.z,
            'height': bbox.h, 'width': bbox.w, 'length': bbox.l, 'heading': bbox.o}
    
    @classmethod
    def bbox2array(cls, bbox):
        if bbox.s is None:
            return np.array([bbox.x, bbox.y, bbox.z, bbox.o, bbox.l, bbox.w, bbox.h])
        else:
            return np.array([bbox.x, bbox.y, bbox.z, bbox.o, bbox.l, bbox.w, bbox.h, bbox.s])

    @classmethod
    def array2bbox(cls, data):
        bbox = BBox()
        bbox.x, bbox.y, bbox.z, bbox.o, bbox.l, bbox.w, bbox.h = data[:7]
        if len(data) == 8:
            bbox.s = data[-1]
        return bbox
    
    @classmethod
    def dict2bbox(cls, data):
        bbox = BBox()
        bbox.x = data['center_x']
        bbox.y = data['center_y']
        bbox.z = data['center_z']
        bbox.h = data['height']
        bbox.w = data['width']
        bbox.l = data['length']
        bbox.o = data['heading']
        if 'score' in data.keys():
            bbox.s = data['score']
        return bbox
    
    @classmethod
    def copy_bbox(cls, bboxa, bboxb):
        bboxa.x = bboxb.x
        bboxa.y = bboxb.y
        bboxa.z = bboxb.z
        bboxa.l = bboxb.l
        bboxa.w = bboxb.w
        bboxa.h = bboxb.h
        bboxa.o = bboxb.o
        bboxa.s = bboxb.s
        return
    
    @classmethod
    def box2corners2d(cls, bbox):
        bottom_center = np.array([bbox.x, bbox.y, bbox.z - bbox.h / 2])
        cos, sin = np.cos(bbox.o), np.sin(bbox.o)
        pc0 = np.array([bbox.x + cos * bbox.l / 2 + sin * bbox.w / 2,
                        bbox.y + sin * bbox.l / 2 - cos * bbox.w / 2,
                        bbox.z - bbox.h / 2])
        pc1 = np.array([bbox.x + cos * bbox.l / 2 - sin * bbox.w / 2,
                        bbox.y + sin * bbox.l / 2 + cos * bbox.w / 2,
                        bbox.z - bbox.h / 2])
        pc2 = 2 * bottom_center - pc0
        pc3 = 2 * bottom_center - pc1
    
        return [pc0.tolist(), pc1.tolist(), pc2.tolist(), pc3.tolist()]
    
    @classmethod
    def box2corners3d(cls, bbox):
        center = np.array([bbox.x, bbox.y, bbox.z])
        bottom_corners = np.array(BBox.box2corners2d(bbox))
        up_corners = 2 * center - bottom_corners
        corners = np.concatenate([up_corners, bottom_corners], axis=0)
        return corners.tolist()
    
    @classmethod
    def motion2bbox(cls, bbox, motion):
        result = deepcopy(bbox)
        result.x += motion[0]
        result.y += motion[1]
        result.z += motion[2]
        result.o += motion[3]
        return result
    
    @classmethod
    def set_bbox_size(cls, bbox, size_array):
        result = deepcopy(bbox)
        result.l, result.w, result.h = size_array
        return result
    
    @classmethod
    def set_bbox_with_states(cls, prev_bbox, state_array):
        prev_array = BBox.bbox2array(prev_bbox)
        prev_array[:4] += state_array[:4]
        prev_array[4:] = state_array[4:]
        bbox = BBox.array2bbox(prev_array)
        return bbox 
    
    @classmethod
    def box_pts2world(cls, ego_matrix, pcs):
        new_pcs = np.concatenate((pcs,
                                  np.ones(pcs.shape[0])[:, np.newaxis]),
                                  axis=1)
        new_pcs = ego_matrix @ new_pcs.T
        new_pcs = new_pcs.T[:, :3]
        return new_pcs
    
    @classmethod
    def edge2yaw(cls, center, edge):
        vec = edge - center
        yaw = np.arccos(vec[0] / np.linalg.norm(vec))
        if vec[1] < 0:
            yaw = -yaw
        return yaw
    
    @classmethod
    def bbox2world(cls, ego_matrix, box):
        # center and corners
        corners = np.array(BBox.box2corners2d(box))
        center = BBox.bbox2array(box)[:3][np.newaxis, :]
        center = BBox.box_pts2world(ego_matrix, center)[0]
        corners = BBox.box_pts2world(ego_matrix, corners)
        # heading
        edge_mid_point = (corners[0] + corners[1]) / 2
        yaw = BBox.edge2yaw(center[:2], edge_mid_point[:2])
        
        result = deepcopy(box)
        result.x, result.y, result.z = center
        result.o = yaw
        return result
__all__ = ['pc_in_box', 'downsample', 'pc_in_box_2D',
           'apply_motion_to_points', 'make_transformation_matrix',
           'iou2d', 'iou3d', 'pc2world', 'giou2d', 'giou3d', 
           'back_step_det', 'm_distance', 'velo2world', 'score_rectification']


def velo2world(ego_matrix, velo):
    new_velo = velo[:, np.newaxis]
    new_velo = ego_matrix[:2, :2] @ new_velo
    return new_velo[:, 0]


def apply_motion_to_points(points, motion, pre_move=0):
    transformation_matrix = make_transformation_matrix(motion)
    points = deepcopy(points)
    points = points + pre_move
    new_points = np.concatenate((points,
                                 np.ones(points.shape[0])[:, np.newaxis]),
                                 axis=1)

    new_points = transformation_matrix @ new_points.T
    new_points = new_points.T[:, :3]
    new_points -= pre_move
    return new_points


@numba.njit
def downsample(points, voxel_size=0.05):
    sample_dict = dict()
    for i in range(points.shape[0]):
        point_coord = np.floor(points[i] / voxel_size)
        sample_dict[(int(point_coord[0]), int(point_coord[1]), int(point_coord[2]))] = True
    res = np.zeros((len(sample_dict), 3), dtype=np.float32)
    idx = 0
    for k, v in sample_dict.items():
        res[idx, 0] = k[0] * voxel_size + voxel_size / 2
        res[idx, 1] = k[1] * voxel_size + voxel_size / 2
        res[idx, 2] = k[2] * voxel_size + voxel_size / 2
        idx += 1
    return res

def pc_in_box(box, pc, box_scaling=1.5):
    center_x, center_y, length, width = \
        box.x, box.y, box.l, box.w
    center_z, height = box.z, box.h
    yaw = box.o
    return pc_in_box_inner(center_x, center_y, center_z, length, width, height, yaw, pc, box_scaling)


@numba.njit
def pc_in_box_inner(center_x, center_y, center_z, length, width, height, yaw, pc, box_scaling=1.5):
    mask = np.zeros(pc.shape[0], dtype=np.int32)
    yaw_cos, yaw_sin = np.cos(yaw), np.sin(yaw)
    for i in range(pc.shape[0]):
        rx = np.abs((pc[i, 0] - center_x) * yaw_cos + (pc[i, 1] - center_y) * yaw_sin)
        ry = np.abs((pc[i, 0] - center_x) * -yaw_sin + (pc[i, 1] - center_y) * yaw_cos)
        rz = np.abs(pc[i, 2] - center_z)

        if rx < (length * box_scaling / 2) and ry < (width * box_scaling / 2) and rz < (height * box_scaling / 2):
            mask[i] = 1
    indices = np.argwhere(mask == 1)
    result = np.zeros((indices.shape[0], 3), dtype=np.float64)
    for i in range(indices.shape[0]):
        result[i, :] = pc[indices[i], :]
    return result


def pc_in_box_2D(box, pc, box_scaling=1.0):
    center_x, center_y, length, width = \
        box.x, box.y, box.l, box.w
    center_z, height = box.z, box.h
    yaw = box.o
    return pc_in_box_2D_inner(center_x, center_y, center_z, length, width, height, yaw, pc, box_scaling)


@numba.njit
def pc_in_box_2D_inner(center_x, center_y, center_z, length, width, height, yaw, pc, box_scaling=1.0):
    mask = np.zeros(pc.shape[0], dtype=np.int32)
    yaw_cos, yaw_sin = np.cos(yaw), np.sin(yaw)
    for i in range(pc.shape[0]):
        rx = np.abs((pc[i, 0] - center_x) * yaw_cos + (pc[i, 1] - center_y) * yaw_sin)
        ry = np.abs((pc[i, 0] - center_x) * -yaw_sin + (pc[i, 1] - center_y) * yaw_cos)

        if rx < (length * box_scaling / 2) and ry < (width * box_scaling / 2):
            mask[i] = 1
    indices = np.argwhere(mask == 1)
    result = np.zeros((indices.shape[0], 3), dtype=np.float64)
    for i in range(indices.shape[0]):
        result[i, :] = pc[indices[i], :]
    return result


def make_transformation_matrix(motion):
    x, y, z, theta = motion
    transformation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0, x],
                                      [np.sin(theta),  np.cos(theta), 0, y],
                                      [0            ,  0            , 1, z],
                                      [0            ,  0            , 0, 1]])
    return transformation_matrix


def iou2d(box_a, box_b):
    boxa_corners = np.array(BBox.box2corners2d(box_a))[:, :2]
    boxb_corners = np.array(BBox.box2corners2d(box_b))[:, :2]
    reca, recb = Polygon(boxa_corners), Polygon(boxb_corners)
    overlap = reca.intersection(recb).area
    area_a = reca.area
    area_b = recb.area
    iou = overlap / (area_a + area_b - overlap + 1e-10)
    return iou


def iou3d(box_a, box_b):
    boxa_corners = np.array(BBox.box2corners2d(box_a))
    boxb_corners = np.array(BBox.box2corners2d(box_b))[:, :2]
    reca, recb = Polygon(boxa_corners), Polygon(boxb_corners)
    overlap_area = reca.intersection(recb).area
    iou_2d = overlap_area / (reca.area + recb.area - overlap_area)

    ha, hb = box_a.h, box_b.h
    za, zb = box_a.z, box_b.z
    overlap_height = max(0, min((za + ha / 2) - (zb - hb / 2), (zb + hb / 2) - (za - ha / 2)))
    overlap_volume = overlap_area * overlap_height
    union_volume = box_a.w * box_a.l * ha + box_b.w * box_b.l * hb - overlap_volume
    iou_3d = overlap_volume / (union_volume + 1e-5)

    return iou_2d, iou_3d


def pc2world(ego_matrix, pcs):
    new_pcs = np.concatenate((pcs,
                              np.ones(pcs.shape[0])[:, np.newaxis]),
                              axis=1)
    new_pcs = ego_matrix @ new_pcs.T
    new_pcs = new_pcs.T[:, :3]
    return new_pcs


def giou2d(box_a: BBox, box_b: BBox):
    boxa_corners = np.array(BBox.box2corners2d(box_a))
    boxb_corners = np.array(BBox.box2corners2d(box_b))
    reca, recb = Polygon(boxa_corners), Polygon(boxb_corners)
    
    I = reca.intersection(recb).area
    U = box_a.w * box_a.l + box_b.w * box_b.l - I

    all_corners = np.vstack((boxa_corners, boxb_corners))
    C = ConvexHull(all_corners)
    convex_corners = all_corners[C.vertices]
    convex_area = PolyArea2D(convex_corners)
    C = convex_area

    return I / U - (C - U) / C
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion

def nu_array2mot_bbox(b):
    translation=b['translation']
    size=b['size']
    rotation=b['rotation']

    nu_box = Box(translation, size, Quaternion(rotation))
    mot_bbox = BBox(
        x=nu_box.center[0], y=nu_box.center[1], z=nu_box.center[2],
        w=nu_box.wlh[0], l=nu_box.wlh[1], h=nu_box.wlh[2],
        o=nu_box.orientation.yaw_pitch_roll[0]
    )
    if 'detection_score' in b.keys():
        mot_bbox.s = b['detection_score']
    if 'tracking_score' in b.keys():
        mot_bbox.s = b['tracking_score']
    return mot_bbox

def giou3d(z, track):
    z=nu_array2mot_bbox(z)
    track=nu_array2mot_bbox(track)

    boxa_corners = np.array(BBox.box2corners2d(z))[:, :2]
    boxb_corners = np.array(BBox.box2corners2d(track))[:, :2]
    reca, recb = Polygon(boxa_corners), Polygon(boxb_corners)
    ha, hb = z.h, track.h
    za, zb = z.z, track.z
    overlap_height = max(0, min((za + ha / 2) - (zb - hb / 2), (zb + hb / 2) - (za - ha / 2)))
    union_height = max((za + ha / 2) - (zb - hb / 2), (zb + hb / 2) - (za - ha / 2))
    
    I = reca.intersection(recb).area * overlap_height
    U = z.w * z.l * ha + track.w * track.l * hb - I

    all_corners = np.vstack((boxa_corners, boxb_corners))
    C = ConvexHull(all_corners)
    convex_corners = all_corners[C.vertices]
    convex_area = PolyArea2D(convex_corners)
    C = convex_area * union_height

    giou = I / U - (C - U) / C
    return giou


def PolyArea2D(pts):
    roll_pts = np.roll(pts, -1, axis=0)
    area = np.abs(np.sum((pts[:, 0] * roll_pts[:, 1] - pts[:, 1] * roll_pts[:, 0]))) * 0.5
    return area


def back_step_det(det: BBox, velo, time_lag):
    result = BBox()
    BBox.copy_bbox(result, det)
    result.x -= (time_lag * velo[0])
    result.y -= (time_lag * velo[1])
    return result


def diff_orientation_correction(diff):
    if diff > np.pi / 2:
        diff -= np.pi
    if diff < -np.pi / 2:
        diff += np.pi
    return diff


def m_distance(det, trk, trk_inv_innovation_matrix=None):
    det_array = BBox.bbox2array(det)[:7]
    trk_array = BBox.bbox2array(trk)[:7]
    
    diff = np.expand_dims(det_array - trk_array, axis=1)
    corrected_yaw_diff = diff_orientation_correction(diff[3])
    diff[3] = corrected_yaw_diff

    if trk_inv_innovation_matrix is not None:
        result = \
            np.sqrt(np.matmul(np.matmul(diff.T, trk_inv_innovation_matrix), diff)[0][0])
    else:
        result = np.sqrt(np.dot(diff.T, diff))
    return result


def score_rectification(dets, gts):
    result = deepcopy(dets)
    
    if len(gts) == 0:
        for i, _ in enumerate(dets):
            result[i].s = 0.0
        return result

    if len(dets) == 0:
        return result

    iou_matrix = np.zeros((len(dets), len(gts)))
    for i, d in enumerate(dets):
        for j, g in enumerate(gts):
            iou_matrix[i, j] = iou3d(d, g)[1]
    max_index = np.argmax(iou_matrix, axis=1)
    max_iou = np.max(iou_matrix, axis=1)
    index = list(reversed(sorted(range(len(dets)), key=lambda k:max_iou[k])))

    matched_gt = []
    for i in index:
        if max_iou[i] >= 0.1 and max_index[i] not in matched_gt:
            result[i].s = max_iou[i]
            matched_gt.append(max_index[i])
        elif max_iou[i] >= 0.1 and max_index[i] in matched_gt:
            result[i].s = 0.2
        else:
            result[i].s = 0.05

    return result


def imagetovideo(image_path, num_images, video_path):
    image_folder = image_path
    images = [img for img in os.listdir(image_folder)]
    frame = cv2.imread(os.path.join(image_folder, images[0]))

    fps = 2

    height, width, layers = frame.shape  

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video = cv2.VideoWriter(video_path, fourcc, fps, (width, height)) 
  
    for i in range(num_images):
        video.write(cv2.imread(os.path.join(image_folder, '{}.png'.format(i)))) 
      
    cv2.destroyAllWindows() 
    video.release()

def generate_video(log_token, out_file_directory_for_this_log):
    num_of_images = len(os.listdir(out_file_directory_for_this_log))
    video_path = out_file_directory_for_this_log+'/{}.mp4'.format(log_token)
    imagetovideo(out_file_directory_for_this_log, num_of_images,video_path)


def generate_visualization(nuscenes_data, root_directory_for_out_path):
    scenes=nuscenes_data.scene
    frames=nuscenes_data.sample
    
    for scene in scenes:
        scene_token=scene['token']
        scene_name=scene['name']
        out_file_directory_for_this_scene = os.path.join(root_directory_for_out_path,scene_name)
        if os.path.exists(out_file_directory_for_this_scene):
            print('erasing existing data')
            shutil.rmtree(out_file_directory_for_this_scene, ignore_errors=True)
        os.mkdir(out_file_directory_for_this_scene)
        
        frames_for_this_scene = []
        for frame in frames:
            if frame['scene_token']==scene_token:
                frames_for_this_scene.append(frame)
        
        unordered_frames = copy.deepcopy(frames_for_this_scene)
        ordered_frames=[]
        while len(unordered_frames)!=0:
            for current_frame in unordered_frames:
                
                if current_frame['prev']=='':
                    ordered_frames.append(current_frame)
                    
                    current_frame_token_of_current_scene = current_frame['token']
                    unordered_frames.remove(current_frame)
        
                
                if current_frame['prev']==current_frame_token_of_current_scene:
                    ordered_frames.append(current_frame)
                    
                    current_frame_token_of_current_scene=current_frame['token']
                    unordered_frames.remove(current_frame)

        
        for idx in range(len(ordered_frames)):
             
            nuscenes_data.render_sample(ordered_frames[idx]['token'],out_path=out_file_directory_for_this_scene+'/{}.png'.format(idx),verbose=False)        
            plt.close('all')
    

def generate_inference_visualization(nuscenes_data,inference_result, nsweeps, root_directory_for_out_path):
    scenes=nuscenes_data.scene
    frames=nuscenes_data.sample
    log=nuscenes_data.log
    for log_file in log:
        log_token = log_file['token']
        out_file_directory_for_this_log = os.path.join(root_directory_for_out_path,log_token)
        
        if os.path.exists(out_file_directory_for_this_log):
            print('Erasing existing data for log {}'.format(log_token))
            shutil.rmtree(out_file_directory_for_this_log, ignore_errors=True)
        os.mkdir(out_file_directory_for_this_log)

    for scene in scenes:
        
        scene_token=scene['token']

        scene_log_token = scene['log_token']
        out_file_directory_for_this_log = os.path.join(root_directory_for_out_path,scene_log_token)

        frames_for_this_scene = []
        for frame in frames:
            if frame['scene_token']==scene_token:
                frames_for_this_scene.append(frame)
        
        unordered_frames = copy.deepcopy(frames_for_this_scene)
        ordered_frames=[]
        
        while len(unordered_frames)!=0:
            for current_frame in unordered_frames:
                
                if current_frame['prev']=='':
                    ordered_frames.append(current_frame)
                    
                    current_frame_token_of_current_scene = current_frame['token']
                    unordered_frames.remove(current_frame)
        
                if current_frame['prev']==current_frame_token_of_current_scene:
                    ordered_frames.append(current_frame)
                    
                    current_frame_token_of_current_scene=current_frame['token']
                    unordered_frames.remove(current_frame)

        num_of_images = len(os.listdir(out_file_directory_for_this_log))

        for idx, frame in enumerate(ordered_frames):
              
            nuscenes_data.render_inference_sample(inference_result,frame['token'],nsweeps=nsweeps,out_path=out_file_directory_for_this_log+'/{}.png'.format(num_of_images+idx),verbose=False)        
            plt.close('all')
    
    for log_file in log:
        log_token = log_file['token']
        out_file_directory_for_this_log = os.path.join(root_directory_for_out_path,log_token)
        generate_video(log_token,out_file_directory_for_this_log)


def get_inference_colormap(class_name):
    classname_to_color = {  # RGB.
        "car": (0, 0, 230),  # Blue
        "truck": (70, 130, 180),  # Steelblue
        "trailer": (138, 43, 226),  # Blueviolet
        "bus":(0,255,0),  # lime
        "construction_vehicle": (255,255,0),  # Gold
        "bicycle": (0, 175, 0),  # Green
        "motorcycle": (0, 0, 128),  # Navy,
        "pedestrian":(255, 69, 0),  # Orangered.
        "traffic_cone": (255,0,255), #magenta
        "barrier": (173,255,47),  # greenyellow
    }

    return classname_to_color[class_name]


def boxes_iou_bev(box_a, box_b):
    translation_of_box_a=[box_a['translation'][0],box_a['translation'][1],box_a['translation'][2]]
    size_of_box_a=box_a['size']
    rotation_of_box_a=Quaternion(box_a['rotation'])
    box_a_in_Box_format = Box(translation_of_box_a,size_of_box_a,rotation_of_box_a)

    translation_of_box_b=[box_b['translation'][0],box_b['translation'][1],box_b['translation'][2]]
    size_of_box_b=box_b['size']
    rotation_of_box_b=Quaternion(box_b['rotation'])
    box_b_in_Box_format = Box(translation_of_box_b,size_of_box_b,rotation_of_box_b)

    x1=box_a_in_Box_format.center[0]
    y1=box_a_in_Box_format.center[1]

    w1=box_a_in_Box_format.wlh[0]
    l1=box_a_in_Box_format.wlh[1]
    
    x2=box_b_in_Box_format.center[0]
    y2=box_b_in_Box_format.center[1]

    w2=box_b_in_Box_format.wlh[0]
    l2=box_b_in_Box_format.wlh[1]

    xA = max(x1, x2)
    yA = max(y1, y2)
    xB = min(x1+w1/2, x2+w2/2)
    yB = min(y1+l1/2, y2+l2/2)

    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0

    boxAArea = abs(w1 * l1)
    boxBArea = abs(w2 * l2)

    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou

def compute_trajectory(frame_index, distance, estimates_previous_frame, estimates_this_frame):
    T=0.5
    I = T*np.eye(2, dtype=np.float64)
    F=np.eye(4, dtype=np.float64)
    F[0:2, 2:4] = I
    sigma_v = 1
    Q1 = np.array([[T ** 4 / 4, T ** 3 / 2], [T ** 3 / 2, T ** 2]], dtype=np.float64)
    Q = np.zeros((4, 4), dtype=np.float64)
    Q[np.ix_([0, 2], [0, 2])] = Q1
    Q[np.ix_([1, 3], [1, 3])] = Q1
    Q = sigma_v ** 2 * Q
    
    if len(estimates_previous_frame['mean'])==0:
        return estimates_this_frame
    else:
        num_previous_estimation = len(estimates_previous_frame['mean'])

        num_current_estimation = len(estimates_this_frame['mean'])
        
        cost_matrix = np.zeros((num_previous_estimation, num_current_estimation))
        for n_previous in range(num_previous_estimation):
            for n_current in range(num_current_estimation):
                if distance=='Euclidean distance':
                    predicted_position=F.dot(estimates_previous_frame['mean'][n_previous])
                    current_cost = (predicted_position[0]-estimates_this_frame['mean'][n_current][0])**2+(predicted_position[1]-estimates_this_frame['mean'][n_current][1])**2
                    cost_matrix[n_previous,n_current] = np.min([current_cost, 20])
            
            previous_frame_assignment, current_frame_assignment = linear_sum_assignment(cost_matrix)
            
            previous_to_current_assigments = dict()
            current_to_previous_assignments=dict()
            
            estimates_this_frame['id']=[-1 for x in range(num_current_estimation)]
            for previous_idx, current_idx in zip(previous_frame_assignment, current_frame_assignment):
                if cost_matrix[previous_idx, current_idx] < 20:
                    previous_to_current_assigments[previous_idx] = current_idx
                    current_to_previous_assignments[current_idx]=previous_idx
                    estimates_this_frame['id'][current_idx]=estimates_previous_frame['id'][previous_idx]
                    estimates_this_frame['classification'][current_idx]=estimates_previous_frame['classification'][previous_idx]
            
            estimates_this_frame['max_id']=estimates_previous_frame['max_id']
            previous_max=estimates_previous_frame['max_id']
            max=previous_max
            for current_index in range(len(estimates_this_frame['mean'])):
                if current_index not in current_to_previous_assignments:
                    max+=1
                    estimates_this_frame['id'][current_index]=max
                    estimates_this_frame['max_id']=max
            
        return estimates_this_frame
    
def readout_parameters(classification, parameters):
    
    parameters_for_this_classification=parameters[classification]
    birth_rate=parameters_for_this_classification['birth_rate']
    P_s=parameters_for_this_classification['p_s']
    P_d=parameters_for_this_classification['p_d']
    use_ds_as_pd=parameters_for_this_classification['use_ds_as_pd']
    clutter_rate=parameters_for_this_classification['clutter_rate']
    bernoulli_gating=parameters_for_this_classification['bernoulli_gating']
    extraction_thr=parameters_for_this_classification['extraction_thr']
    ber_thr=parameters_for_this_classification['ber_thr']
    poi_thr=parameters_for_this_classification['poi_thr']
    eB_thr=parameters_for_this_classification['eB_thr']
    detection_score_thr=parameters_for_this_classification['detection_score_thr']
    nms_score = parameters_for_this_classification['nms_score']
    confidence_score = parameters_for_this_classification['confidence_score']
    P_init = parameters_for_this_classification['P_init']
    return birth_rate, P_s,P_d, use_ds_as_pd, clutter_rate, bernoulli_gating, extraction_thr, ber_thr, poi_thr, eB_thr, detection_score_thr, nms_score, confidence_score, P_init

def readout_gnn_parameters(classification, parameters):
    
    parameters_for_this_classification=parameters[classification]
    gating=parameters_for_this_classification['gating']
    P_d=parameters_for_this_classification['p_d']
    clutter_rate=parameters_for_this_classification['clutter_rate']
    detection_score_thr=parameters_for_this_classification['detection_score_thr']
    nms_score = parameters_for_this_classification['nms_score']
    death_counter_kill=parameters_for_this_classification['death_counter_kill']
    birth_counter_born=parameters_for_this_classification['birth_counter_born']
    death_initiation=parameters_for_this_classification['death_initiation']
    birth_initiation=parameters_for_this_classification['birth_initiation']
    return gating,P_d, clutter_rate,detection_score_thr, nms_score, death_counter_kill,birth_counter_born,death_initiation,birth_initiation


def initiate_submission_file(orderedframe):
    submission={}
    submission['meta']={}
    submission['meta']["use_camera"]=False
    submission['meta']["use_lidar"]=True
    submission['meta']["use_radar"]=False
    submission['meta']["use_map"]=False
    submission['meta']["use_external"]=False
    submission['results']={}
    for scene_token in orderedframe.keys():
        frames=orderedframe[scene_token]
        for frame_token in frames:
            submission['results'][frame_token]=[]
    return submission

def initiate_submission_file_mini(frames,estimated_bboxes_data_over_all_frames):
    submission={}
    submission['meta']={}
    submission['meta']["use_camera"]=False
    submission['meta']["use_lidar"]=True
    submission['meta']["use_radar"]=False
    submission['meta']["use_map"]=False
    submission['meta']["use_external"]=False
    submission['results']={}
    for frame in frames:
        frame_token=frame['token']
        if frame_token in estimated_bboxes_data_over_all_frames.keys():
            submission['results'][frame_token]=[]
    return submission

def initiate_classification_submission_file(classification):
    submission={}
    submission['meta']={}
    submission['meta']["use_camera"]=False
    submission['meta']["use_lidar"]=True
    submission['meta']["use_radar"]=False
    submission['meta']["use_map"]=False
    submission['meta']["use_external"]=False
    submission['meta']["classification"]=classification
    submission['results']={}
    return submission

def create_experiment_folder(root_directory_for_dataset, time,comment):
    result_path=os.path.join(root_directory_for_dataset, 'experiment_result')
    
    timefolder=os.path.join(result_path,time)
    experiment_folder=timefolder+'_'+comment
    
    if os.path.exists(experiment_folder):
        pass
    else:
        os.makedirs(experiment_folder)
    return experiment_folder
def create_scene_folder(scene, experiment_folder):
    
    scene_token=scene['token']
    
    scene_name=scene['name']
    
    out_file_directory_for_this_scene = os.path.join(experiment_folder,scene_name)
    
    if os.path.exists(out_file_directory_for_this_scene):
        pass
    else:
        os.mkdir(out_file_directory_for_this_scene)
    return out_file_directory_for_this_scene

def create_classification_folder(classification,scene_folder):
    out_file_directory_for_this_scene_classfication = os.path.join(scene_folder, classification)
    
    if os.path.exists(out_file_directory_for_this_scene_classfication):
        pass
    else:
        os.mkdir(out_file_directory_for_this_scene_classfication)
    return out_file_directory_for_this_scene_classfication

def gen_ordered_frames(scene,frames):
    
    frames_for_this_scene = []
    for frame in frames:
        if frame['scene_token']==scene['token']:
            frames_for_this_scene.append(frame)
            unordered_frames = copy.deepcopy(frames_for_this_scene)
            ordered_frames=[]
            
            while len(unordered_frames)!=0:
                for current_frame in unordered_frames:
                    
                    if current_frame['prev']=='':
                        ordered_frames.append(current_frame)
                        
                        current_frame_token_of_current_scene = current_frame['token']
                        unordered_frames.remove(current_frame)
            
                    if current_frame['prev']==current_frame_token_of_current_scene:
                        ordered_frames.append(current_frame)
                        
                        current_frame_token_of_current_scene=current_frame['token']
                        unordered_frames.remove(current_frame)
    return ordered_frames

def gen_measurement_of_this_class(detection_score_thr,estimated_bboxes_at_current_frame, classification):
    Z_k=[]
    for box_index, box in enumerate(estimated_bboxes_at_current_frame):
        if box['detection_name']==classification:
            if box['detection_score']>detection_score_thr:
                Z_k.append(box)
    return Z_k


def instance_info2bbox_array(info):
    translation = info.center.tolist()
    size = info.wlh.tolist()
    rotation = info.orientation.q.tolist()
    return translation + size + rotation

class BBoxCoarseFilter:
    def __init__(self, grid_size, scaler=100):
        self.gsize = grid_size
        self.scaler = 100
        self.bbox_dict = dict()
    
    def bboxes2dict(self, bboxes):
        for i, bbox in enumerate(bboxes):
            grid_keys = self.compute_bbox_key(bbox)
            for key in grid_keys:
                if key not in self.bbox_dict.keys():
                    self.bbox_dict[key] = set([i])
                else:
                    self.bbox_dict[key].add(i)
        return
        
    def compute_bbox_key(self, bbox):
        corners = np.asarray(BBox.box2corners2d(bbox))
        min_keys = np.floor(np.min(corners, axis=0) / self.gsize).astype(np.int)
        max_keys = np.floor(np.max(corners, axis=0) / self.gsize).astype(np.int)
        
        # enumerate all the corners
        grid_keys = [
            self.scaler * min_keys[0] + min_keys[1],
            self.scaler * min_keys[0] + max_keys[1],
            self.scaler * max_keys[0] + min_keys[1],
            self.scaler * max_keys[0] + max_keys[1]
        ]
        return grid_keys
    
    def related_bboxes(self, bbox):
        result = set()
        grid_keys = self.compute_bbox_key(bbox)
        for key in grid_keys:
            if key in self.bbox_dict.keys():
                result.update(self.bbox_dict[key])
        return list(result)
    
    def clear(self):
        self.bbox_dict = dict()

def weird_bbox(bbox):
    if bbox.l <= 0 or bbox.w <= 0 or bbox.h <= 0:
        return True
    else:
        return False

def nms(dets, threshold=0.1, threshold_high=1.0, threshold_yaw=0.3):
    dets_new=[]
    for det in dets:
        dets_new.append(nu_array2mot_bbox(det))
    dets=dets_new

    dets_coarse_filter = BBoxCoarseFilter(grid_size=100, scaler=100)
    dets_coarse_filter.bboxes2dict(dets)
    scores = np.asarray([det.s for det in dets])
    yaws = np.asarray([det.o for det in dets])
    order = np.argsort(scores)[::-1]
    
    result_indexes = list()
    while order.size > 0:
        index = order[0]

        if weird_bbox(dets[index]):
            order = order[1:]
            continue

        filter_indexes = dets_coarse_filter.related_bboxes(dets[index])
        in_mask = np.isin(order, filter_indexes)
        related_idxes = order[in_mask]

        bbox_num = len(related_idxes)
        ious = np.zeros(bbox_num)
        for i, idx in enumerate(related_idxes):
            ious[i] = iou3d(dets[index], dets[idx])[1]
        related_inds = np.where(ious > threshold)
        related_inds_vote = np.where(ious > threshold_high)
        order_vote = related_idxes[related_inds_vote]

        if len(order_vote) >= 2:
            
            if order_vote.shape[0] <= 2:
                score_index = np.argmax(scores[order_vote])
                median_yaw = yaws[order_vote][score_index]
            elif order_vote.shape[0] % 2 == 0:
                tmp_yaw = yaws[order_vote].copy()
                tmp_yaw = np.append(tmp_yaw, yaws[order_vote][0])
                median_yaw = np.median(tmp_yaw)
            else:
                median_yaw = np.median(yaws[order_vote])
            yaw_vote = np.where(np.abs(yaws[order_vote] - median_yaw) % (2 * np.pi) < threshold_yaw)[0]
            order_vote = order_vote[yaw_vote]
            
            vote_score_sum = np.sum(scores[order_vote])
            det_arrays = list()
            for idx in order_vote:
                det_arrays.append(BBox.bbox2array(dets[idx])[np.newaxis, :])
            det_arrays = np.vstack(det_arrays)
            avg_bbox_array = np.sum(scores[order_vote][:, np.newaxis] * det_arrays, axis=0) / vote_score_sum
            bbox = BBox.array2bbox(avg_bbox_array)
            bbox.s = scores[index]
            result_indexes.append(index)
        else:
            result_indexes.append(index)

        delete_idxes = related_idxes[related_inds]
        in_mask = np.isin(order, delete_idxes, invert=True)
        order = order[in_mask]

    return result_indexes


def associate_dets_to_tracks(dets, tracks, mode, asso, 
    dist_threshold=0.9, trk_innovation_matrix=None):

    if mode == 'bipartite':
        matched_indices, dist_matrix = \
            bipartite_matcher(dets, tracks, asso, dist_threshold, trk_innovation_matrix)
    elif mode == 'greedy':
        matched_indices, dist_matrix = \
            greedy_matcher(dets, tracks, asso, dist_threshold, trk_innovation_matrix)
    unmatched_dets = list()
    for d, det in enumerate(dets):
        if d not in matched_indices[:, 0]:
            unmatched_dets.append(d)

    unmatched_tracks = list()
    for t, trk in enumerate(tracks):
        if t not in matched_indices[:, 1]:
            unmatched_tracks.append(t)
    
    matches = list()
    for m in matched_indices:
        if dist_matrix[m[0], m[1]] > dist_threshold:
            unmatched_dets.append(m[0])
            unmatched_tracks.append(m[1])
        else:
            matches.append(m.reshape(2))
    return matches, np.array(unmatched_dets), np.array(unmatched_tracks)


def bipartite_matcher(dets, tracks, asso, dist_threshold, trk_innovation_matrix):
    if asso == 'iou':
        dist_matrix = compute_iou_distance(dets, tracks, asso)
    elif asso == 'giou':
        dist_matrix = compute_iou_distance(dets, tracks, asso)
    elif asso == 'm_dis':
        dist_matrix = compute_m_distance(dets, tracks, trk_innovation_matrix)
    elif asso == 'euler':
        dist_matrix = compute_m_distance(dets, tracks, None)
    row_ind, col_ind = linear_sum_assignment(dist_matrix)
    matched_indices = np.stack([row_ind, col_ind], axis=1)
    return matched_indices, dist_matrix


def greedy_matcher(dets, tracks, asso, dist_threshold, trk_innovation_matrix):
    matched_indices = list()
    
    if asso == 'm_dis':
        distance_matrix = compute_m_distance(dets, tracks, trk_innovation_matrix)
    elif asso == 'euler':
        distance_matrix = compute_m_distance(dets, tracks, None)
    elif asso == 'iou':
        distance_matrix = compute_iou_distance(dets, tracks, asso)
    elif asso == 'giou':
        distance_matrix = compute_iou_distance(dets, tracks, asso)
    num_dets, num_trks = distance_matrix.shape

    # refer to https://github.com/eddyhkchiu/mahalanobis_3d_multi_object_tracking/blob/master/main.py
    distance_1d = distance_matrix.reshape(-1)
    index_1d = np.argsort(distance_1d)
    index_2d = np.stack([index_1d // num_trks, index_1d % num_trks], axis=1)
    detection_id_matches_to_tracking_id = [-1] * num_dets
    tracking_id_matches_to_detection_id = [-1] * num_trks
    for sort_i in range(index_2d.shape[0]):
        detection_id = int(index_2d[sort_i][0])
        tracking_id = int(index_2d[sort_i][1])
        if tracking_id_matches_to_detection_id[tracking_id] == -1 and detection_id_matches_to_tracking_id[detection_id] == -1:
            tracking_id_matches_to_detection_id[tracking_id] = detection_id
            detection_id_matches_to_tracking_id[detection_id] = tracking_id
            matched_indices.append([detection_id, tracking_id])
    if len(matched_indices) == 0:
        matched_indices = np.empty((0, 2))
    else:
        matched_indices = np.asarray(matched_indices)
    return matched_indices, distance_matrix


def compute_m_distance(dets, tracks, trk_innovation_matrix):
    euler_dis = (trk_innovation_matrix is None) # is use euler distance
    if not euler_dis:
        trk_inv_inn_matrices = [np.linalg.inv(m) for m in trk_innovation_matrix]
    dist_matrix = np.empty((len(dets), len(tracks)))

    for i, det in enumerate(dets):
        for j, trk in enumerate(tracks):
            if euler_dis:
                dist_matrix[i, j] = m_distance(det, trk)
            else:
                dist_matrix[i, j] = m_distance(det, trk, trk_inv_inn_matrices[j])
    return dist_matrix


def compute_iou_distance(dets, tracks, asso='iou'):
    iou_matrix = np.zeros((len(dets), len(tracks)))
    for d, det in enumerate(dets):
        for t, trk in enumerate(tracks):
            if asso == 'iou':
                iou_matrix[d, t] = iou3d(det, trk)[1]
            elif asso == 'giou':
                iou_matrix[d, t] = giou3d(det, trk)
    dist_matrix = 1 - iou_matrix
    return dist_matrix


def weird_bbox(bbox):
    if bbox.l <= 0 or bbox.w <= 0 or bbox.h <= 0:
        return True
    else:
        return False

class BBoxCoarseFilter:
    def __init__(self, grid_size, scaler=100):
        self.gsize = grid_size
        self.scaler = 100
        self.bbox_dict = dict()
    
    def bboxes2dict(self, bboxes):
        for i, bbox in enumerate(bboxes):
            grid_keys = self.compute_bbox_key(bbox)
            for key in grid_keys:
                if key not in self.bbox_dict.keys():
                    self.bbox_dict[key] = set([i])
                else:
                    self.bbox_dict[key].add(i)
        return
        
    def compute_bbox_key(self, bbox):
        corners = np.asarray(BBox.box2corners2d(bbox))
        min_keys = np.floor(np.min(corners, axis=0) / self.gsize).astype(np.int)
        max_keys = np.floor(np.max(corners, axis=0) / self.gsize).astype(np.int)
        
        # enumerate all the corners
        grid_keys = [
            self.scaler * min_keys[0] + min_keys[1],
            self.scaler * min_keys[0] + max_keys[1],
            self.scaler * max_keys[0] + min_keys[1],
            self.scaler * max_keys[0] + max_keys[1]
        ]
        return grid_keys
    
    def related_bboxes(self, bbox):
        """ return the list of related bboxes
        """ 
        result = set()
        grid_keys = self.compute_bbox_key(bbox)
        for key in grid_keys:
            if key in self.bbox_dict.keys():
                result.update(self.bbox_dict[key])
        return list(result)
    
    def clear(self):
        self.bbox_dict = dict()