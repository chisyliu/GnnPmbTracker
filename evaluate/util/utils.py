# the following code is part of nuScenes dev-kit.
# we put it here instead of use the nuscenes repo is because we adapted the original code for our purpose
# Code written by Holger Caesar, 2019.

# this is the support functions for evaluation
# for detailed explaination of each function
# please see the note for the functions
from functools import reduce
import unittest
from bisect import bisect
import warnings
from abc import ABC, abstractmethod
from typing import Optional, Dict,List, DefaultDict
import abc
from collections import defaultdict
import numpy as np
import sklearn
import os
import copy
from matplotlib import pyplot as plt
import os.path as osp
try:
    import motmetrics
    from motmetrics.metrics import MetricsHost
except ModuleNotFoundError:
    raise unittest.SkipTest('Skipping test as motmetrics was not found!')
import json
import os
from typing import List, Dict, Callable, Tuple, Union
import unittest

import numpy as np
import sklearn

try:
    import pandas
except ModuleNotFoundError:
    raise unittest.SkipTest('Skipping test as pandas was not found!')
from collections import OrderedDict
from itertools import count
import motmetrics
import numpy as np
from matplotlib.axes import Axes
import pandas as pd
from enum import IntEnum
import os
from typing import Any, List
import numpy as np
from pandas import DataFrame
from pyquaternion import Quaternion
from nuscenes.lidarseg.lidarseg_utils import colormap_to_colors, create_lidarseg_legend
from nuscenes.utils.data_io import load_bin_file
from nuscenes.utils.geometry_utils import view_points, transform_matrix
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes

Axis = Any

AMOT_METRICS = ['amota', 'amotp']
INTERNAL_METRICS = ['recall', 'motar', 'gt']
LEGACY_METRICS = ['mota', 'motp', 'mt', 'ml', 'faf', 'tp', 'fp', 'fn', 'ids', 'frag', 'tid', 'lgd']
TRACKING_METRICS = [*AMOT_METRICS, *INTERNAL_METRICS, *LEGACY_METRICS]

# Define mapping for metrics averaged over classes.
AVG_METRIC_MAP = {  # Mapping from average metric name to individual per-threshold metric name.
    'amota': 'motar',
    'amotp': 'motp'
}

# Define mapping for metrics that use motmetrics library.
MOT_METRIC_MAP = {  # Mapping from motmetrics names to metric names used here.
    'num_frames': '',  # Used in FAF.
    'num_objects': 'gt',  # Used in MOTAR computation.
    'num_predictions': '',  # Only printed out.
    'num_matches': 'tp',  # Used in MOTAR computation and printed out.
    'motar': 'motar',  # Only used in AMOTA.
    'mota_custom': 'mota',  # Traditional MOTA, but clipped below 0.
    'motp_custom': 'motp',  # Traditional MOTP.
    'faf': 'faf',
    'mostly_tracked': 'mt',
    'mostly_lost': 'ml',
    'num_false_positives': 'fp',
    'num_misses': 'fn',
    'num_switches': 'ids',
    'num_fragmentations_custom': 'frag',
    'recall': 'recall',
    'tid': 'tid',
    'lgd': 'lgd'
}


DETECTION_NAMES = ['car', 'truck', 'bus', 'trailer', 'construction_vehicle', 'pedestrian', 'motorcycle', 'bicycle',
                   'traffic_cone', 'barrier']

PRETTY_DETECTION_NAMES = {'car': 'Car',
                          'truck': 'Truck',
                          'bus': 'Bus',
                          'trailer': 'Trailer',
                          'construction_vehicle': 'Constr. Veh.',
                          'pedestrian': 'Pedestrian',
                          'motorcycle': 'Motorcycle',
                          'bicycle': 'Bicycle',
                          'traffic_cone': 'Traffic Cone',
                          'barrier': 'Barrier'}

DETECTION_COLORS = {'car': 'C0',
                    'truck': 'C1',
                    'bus': 'C2',
                    'trailer': 'C3',
                    'construction_vehicle': 'C4',
                    'pedestrian': 'C5',
                    'motorcycle': 'C6',
                    'bicycle': 'C7',
                    'traffic_cone': 'C8',
                    'barrier': 'C9'}

ATTRIBUTE_NAMES = ['pedestrian.moving', 'pedestrian.sitting_lying_down', 'pedestrian.standing', 'cycle.with_rider',
                   'cycle.without_rider', 'vehicle.moving', 'vehicle.parked', 'vehicle.stopped']

PRETTY_ATTRIBUTE_NAMES = {'pedestrian.moving': 'Ped. Moving',
                          'pedestrian.sitting_lying_down': 'Ped. Sitting',
                          'pedestrian.standing': 'Ped. Standing',
                          'cycle.with_rider': 'Cycle w/ Rider',
                          'cycle.without_rider': 'Cycle w/o Rider',
                          'vehicle.moving': 'Veh. Moving',
                          'vehicle.parked': 'Veh. Parked',
                          'vehicle.stopped': 'Veh. Stopped'}

TP_METRICS = ['trans_err', 'scale_err', 'orient_err', 'vel_err', 'attr_err']

PRETTY_TP_METRICS = {'trans_err': 'Trans.', 'scale_err': 'Scale', 'orient_err': 'Orient.', 'vel_err': 'Vel.',
                     'attr_err': 'Attr.'}

TP_METRICS_UNITS = {'trans_err': 'm',
                    'scale_err': '1-IOU',
                    'orient_err': 'rad.',
                    'vel_err': 'm/s',
                    'attr_err': '1-acc.'}


class BoxVisibility(IntEnum):
    """ Enumerates the various level of box visibility in an image """
    ALL = 0  # Requires all corners are inside the image.
    ANY = 1  # Requires at least one corner visible in the image.
    NONE = 2  # Requires no corners to be inside, i.e. box can be fully outside the image.


def view_points(points: np.ndarray, view: np.ndarray, normalize: bool) -> np.ndarray:
    """
    This is a helper class that maps 3d points to a 2d plane. It can be used to implement both perspective and
    orthographic projections. It first applies the dot product between the points and the view. By convention,
    the view should be such that the data is projected onto the first 2 axis. It then optionally applies a
    normalization along the third dimension.

    For a perspective projection the view should be a 3x3 camera matrix, and normalize=True
    For an orthographic projection with translation the view is a 3x4 matrix and normalize=False
    For an orthographic projection without translation the view is a 3x3 matrix (optionally 3x4 with last columns
     all zeros) and normalize=False

    :param points: <np.float32: 3, n> Matrix of points, where each point (x, y, z) is along each column.
    :param view: <np.float32: n, n>. Defines an arbitrary projection (n <= 4).
        The projection should be such that the corners are projected onto the first 2 axis.
    :param normalize: Whether to normalize the remaining coordinate (along the third axis).
    :return: <np.float32: 3, n>. Mapped point. If normalize=False, the third coordinate is the height.
    """

    assert view.shape[0] <= 4
    assert view.shape[1] <= 4
    assert points.shape[0] == 3

    viewpad = np.eye(4)
    viewpad[:view.shape[0], :view.shape[1]] = view

    nbr_points = points.shape[1]

    # Do operation in homogenous coordinates.
    points = np.concatenate((points, np.ones((1, nbr_points))))
    points = np.dot(viewpad, points)
    points = points[:3, :]

    if normalize:
        points = points / points[2:3, :].repeat(3, 0).reshape(3, nbr_points)

    return points

def box_in_image(box, intrinsic: np.ndarray, imsize: Tuple[int, int], vis_level: int = BoxVisibility.ANY) -> bool:
    """
    Check if a box is visible inside an image without accounting for occlusions.
    :param box: The box to be checked.
    :param intrinsic: <float: 3, 3>. Intrinsic camera matrix.
    :param imsize: (width, height).
    :param vis_level: One of the enumerations of <BoxVisibility>.
    :return True if visibility condition is satisfied.
    """

    corners_3d = box.corners()
    corners_img = view_points(corners_3d, intrinsic, normalize=True)[:2, :]

    visible = np.logical_and(corners_img[0, :] > 0, corners_img[0, :] < imsize[0])
    visible = np.logical_and(visible, corners_img[1, :] < imsize[1])
    visible = np.logical_and(visible, corners_img[1, :] > 0)
    visible = np.logical_and(visible, corners_3d[2, :] > 1)

    in_front = corners_3d[2, :] > 0.1  # True if a corner is at least 0.1 meter in front of the camera.

    if vis_level == BoxVisibility.ALL:
        return all(visible) and all(in_front)
    elif vis_level == BoxVisibility.ANY:
        return any(visible) and all(in_front)
    elif vis_level == BoxVisibility.NONE:
        return True
    else:
        raise ValueError("vis_level: {} not valid".format(vis_level))


def transform_matrix(translation: np.ndarray = np.array([0, 0, 0]),
                     rotation: Quaternion = Quaternion([1, 0, 0, 0]),
                     inverse: bool = False) -> np.ndarray:
    """
    Convert pose to transformation matrix.
    :param translation: <np.float32: 3>. Translation in x, y, z.
    :param rotation: Rotation in quaternions (w ri rj rk).
    :param inverse: Whether to compute inverse transform matrix.
    :return: <np.float32: 4, 4>. Transformation matrix.
    """
    tm = np.eye(4)

    if inverse:
        rot_inv = rotation.rotation_matrix.T
        trans = np.transpose(-np.array(translation))
        tm[:3, :3] = rot_inv
        tm[:3, 3] = rot_inv.dot(trans)
    else:
        tm[:3, :3] = rotation.rotation_matrix
        tm[:3, 3] = np.transpose(np.array(translation))

    return tm

def points_in_box(box: 'Box', points: np.ndarray, wlh_factor: float = 1.0):
    """
    Checks whether points are inside the box.

    Picks one corner as reference (p1) and computes the vector to a target point (v).
    Then for each of the 3 axes, project v onto the axis and compare the length.
    Inspired by: https://math.stackexchange.com/a/1552579
    :param box: <Box>.
    :param points: <np.float: 3, n>.
    :param wlh_factor: Inflates or deflates the box.
    :return: <np.bool: n, >.
    """
    corners = box.corners(wlh_factor=wlh_factor)

    p1 = corners[:, 0]
    p_x = corners[:, 4]
    p_y = corners[:, 1]
    p_z = corners[:, 3]

    i = p_x - p1
    j = p_y - p1
    k = p_z - p1

    v = points - p1.reshape((-1, 1))

    iv = np.dot(i, v)
    jv = np.dot(j, v)
    kv = np.dot(k, v)

    mask_x = np.logical_and(0 <= iv, iv <= np.dot(i, i))
    mask_y = np.logical_and(0 <= jv, jv <= np.dot(j, j))
    mask_z = np.logical_and(0 <= kv, kv <= np.dot(k, k))
    mask = np.logical_and(np.logical_and(mask_x, mask_y), mask_z)

    return mask

class Box:
    """ Simple data class representing a 3d box including, label, score and velocity. """

    def __init__(self,
                 center: List[float],
                 size: List[float],
                 orientation: Quaternion,
                 label: int = np.nan,
                 score: float = np.nan,
                 velocity: Tuple = (np.nan, np.nan, np.nan),
                 name: str = None,
                 token: str = None):
        """
        :param center: Center of box given as x, y, z.
        :param size: Size of box in width, length, height.
        :param orientation: Box orientation.
        :param label: Integer label, optional.
        :param score: Classification score, optional.
        :param velocity: Box velocity in x, y, z direction.
        :param name: Box name, optional. Can be used e.g. for denote category name.
        :param token: Unique string identifier from DB.
        """
        assert not np.any(np.isnan(center))
        assert not np.any(np.isnan(size))
        assert len(center) == 3
        assert len(size) == 3
        assert type(orientation) == Quaternion

        self.center = np.array(center)
        self.wlh = np.array(size)
        self.orientation = orientation
        self.label = int(label) if not np.isnan(label) else label
        self.score = float(score) if not np.isnan(score) else score
        self.velocity = np.array(velocity)
        self.name = name
        self.token = token

    def __eq__(self, other):
        center = np.allclose(self.center, other.center)
        wlh = np.allclose(self.wlh, other.wlh)
        orientation = np.allclose(self.orientation.elements, other.orientation.elements)
        label = (self.label == other.label) or (np.isnan(self.label) and np.isnan(other.label))
        score = (self.score == other.score) or (np.isnan(self.score) and np.isnan(other.score))
        vel = (np.allclose(self.velocity, other.velocity) or
               (np.all(np.isnan(self.velocity)) and np.all(np.isnan(other.velocity))))

        return center and wlh and orientation and label and score and vel

    def __repr__(self):
        repr_str = 'label: {}, score: {:.2f}, xyz: [{:.2f}, {:.2f}, {:.2f}], wlh: [{:.2f}, {:.2f}, {:.2f}], ' \
                   'rot axis: [{:.2f}, {:.2f}, {:.2f}], ang(degrees): {:.2f}, ang(rad): {:.2f}, ' \
                   'vel: {:.2f}, {:.2f}, {:.2f}, name: {}, token: {}'

        return repr_str.format(self.label, self.score, self.center[0], self.center[1], self.center[2], self.wlh[0],
                               self.wlh[1], self.wlh[2], self.orientation.axis[0], self.orientation.axis[1],
                               self.orientation.axis[2], self.orientation.degrees, self.orientation.radians,
                               self.velocity[0], self.velocity[1], self.velocity[2], self.name, self.token)

    @property
    def rotation_matrix(self) -> np.ndarray:
        """
        Return a rotation matrix.
        :return: <np.float: 3, 3>. The box's rotation matrix.
        """
        return self.orientation.rotation_matrix

    def translate(self, x: np.ndarray) -> None:
        """
        Applies a translation.
        :param x: <np.float: 3, 1>. Translation in x, y, z direction.
        """
        self.center += x

    def rotate(self, quaternion: Quaternion) -> None:
        """
        Rotates box.
        :param quaternion: Rotation to apply.
        """
        self.center = np.dot(quaternion.rotation_matrix, self.center)
        self.orientation = quaternion * self.orientation
        self.velocity = np.dot(quaternion.rotation_matrix, self.velocity)

    def corners(self, wlh_factor: float = 1.0) -> np.ndarray:
        """
        Returns the bounding box corners.
        :param wlh_factor: Multiply w, l, h by a factor to scale the box.
        :return: <np.float: 3, 8>. First four corners are the ones facing forward.
            The last four are the ones facing backwards.
        """
        w, l, h = self.wlh * wlh_factor

        # 3D bounding box corners. (Convention: x points forward, y to the left, z up.)
        x_corners = l / 2 * np.array([1,  1,  1,  1, -1, -1, -1, -1])
        y_corners = w / 2 * np.array([1, -1, -1,  1,  1, -1, -1,  1])
        z_corners = h / 2 * np.array([1,  1, -1, -1,  1,  1, -1, -1])
        corners = np.vstack((x_corners, y_corners, z_corners))

        # Rotate
        corners = np.dot(self.orientation.rotation_matrix, corners)

        # Translate
        x, y, z = self.center
        corners[0, :] = corners[0, :] + x
        corners[1, :] = corners[1, :] + y
        corners[2, :] = corners[2, :] + z

        return corners

    def bottom_corners(self) -> np.ndarray:
        """
        Returns the four bottom corners.
        :return: <np.float: 3, 4>. Bottom corners. First two face forward, last two face backwards.
        """
        return self.corners()[:, [2, 3, 7, 6]]

    def render(self,
               axis: Axes,
               view: np.ndarray = np.eye(3),
               normalize: bool = False,
               colors: Tuple = ('b', 'r', 'k'),
               linewidth: float = 2,
               linestyle: str = '-') -> None:
        """
        Renders the box in the provided Matplotlib axis.
        :param axis: Axis onto which the box should be drawn.
        :param view: <np.array: 3, 3>. Define a projection in needed (e.g. for drawing projection in an image).
        :param normalize: Whether to normalize the remaining coordinate.
        :param colors: (<Matplotlib.colors>: 3). Valid Matplotlib colors (<str> or normalized RGB tuple) for front,
            back and sides.
        :param linewidth: Width in pixel of the box sides.
        """
        corners = view_points(self.corners(), view, normalize=normalize)[:2, :]

        def draw_rect(selected_corners, color):
            prev = selected_corners[-1]
            for corner in selected_corners:
                axis.plot([prev[0], corner[0]], [prev[1], corner[1]], color=color, linewidth=linewidth,linestyle=linestyle)
                prev = corner

        # Draw the sides
        for i in range(4):
            axis.plot([corners.T[i][0], corners.T[i + 4][0]],
                      [corners.T[i][1], corners.T[i + 4][1]],
                      color=colors[2], linewidth=linewidth, linestyle=linestyle)

        # Draw front (first 4 corners) and rear (last 4 corners) rectangles(3d)/lines(2d)
        draw_rect(corners.T[:4], colors[0])
        draw_rect(corners.T[4:], colors[1])

        # Draw line indicating the front
        center_bottom_forward = np.mean(corners.T[2:4], axis=0)
        center_bottom = np.mean(corners.T[[2, 3, 7, 6]], axis=0)
        axis.plot([center_bottom[0], center_bottom_forward[0]],
                  [center_bottom[1], center_bottom_forward[1]],
                  color=colors[0], linewidth=linewidth,linestyle=linestyle)
        return center_bottom

    def copy(self) -> 'Box':
        """
        Create a copy of self.
        :return: A copy.
        """
        return copy.deepcopy(self)


class EvalBox(abc.ABC):
    """ Abstract base class for data classes used during detection evaluation. Can be a prediction or ground truth."""

    def __init__(self,
                 sample_token: str = "",
                 translation: Tuple[float, float, float] = (0, 0, 0),
                 size: Tuple[float, float, float] = (0, 0, 0),
                 rotation: Tuple[float, float, float, float] = (0, 0, 0, 0),
                 velocity: Tuple[float, float] = (0, 0),
                 ego_translation: Tuple[float, float, float] = (0, 0, 0),  # Translation to ego vehicle in meters.
                 num_pts: int = -1):  # Nbr. LIDAR or RADAR inside the box. Only for gt boxes.

        # Assert data for shape and NaNs.
        assert type(sample_token) == str, 'Error: sample_token must be a string!'

        assert len(translation) == 3, 'Error: Translation must have 3 elements!'
        assert not np.any(np.isnan(translation)), 'Error: Translation may not be NaN!'

        assert len(size) == 3, 'Error: Size must have 3 elements!'
        assert not np.any(np.isnan(size)), 'Error: Size may not be NaN!'

        assert len(rotation) == 4, 'Error: Rotation must have 4 elements!'
        assert not np.any(np.isnan(rotation)), 'Error: Rotation may not be NaN!'

        # Velocity can be NaN from our database for certain annotations.
        assert len(velocity) == 2, 'Error: Velocity must have 2 elements!'

        assert len(ego_translation) == 3, 'Error: Translation must have 3 elements!'
        assert not np.any(np.isnan(ego_translation)), 'Error: Translation may not be NaN!'

        assert type(num_pts) == int, 'Error: num_pts must be int!'
        assert not np.any(np.isnan(num_pts)), 'Error: num_pts may not be NaN!'

        # Assign.
        self.sample_token = sample_token
        self.translation = translation
        self.size = size
        self.rotation = rotation
        self.velocity = velocity
        self.ego_translation = ego_translation
        self.num_pts = num_pts

    @property
    def ego_dist(self) -> float:
        """ Compute the distance from this box to the ego vehicle in 2D. """
        return np.sqrt(np.sum(np.array(self.ego_translation[:2]) ** 2))

    def __repr__(self):
        return str(self.serialize())

    @abc.abstractmethod
    def serialize(self) -> dict:
        pass

    @classmethod
    @abc.abstractmethod
    def deserialize(cls, content: dict):
        pass


EvalBoxType = Union['DetectionBox', 'TrackingBox']

class EvalBoxes:
    """ Data class that groups EvalBox instances by sample. """

    def __init__(self):
        """
        Initializes the EvalBoxes for GT or predictions.
        """
        self.boxes = defaultdict(list)

    def __repr__(self):
        return "EvalBoxes with {} boxes across {} samples".format(len(self.all), len(self.sample_tokens))

    def __getitem__(self, item) -> List[EvalBoxType]:
        return self.boxes[item]

    def __eq__(self, other):
        if not set(self.sample_tokens) == set(other.sample_tokens):
            return False
        for token in self.sample_tokens:
            if not len(self[token]) == len(other[token]):
                return False
            for box1, box2 in zip(self[token], other[token]):
                if box1 != box2:
                    return False
        return True

    def __len__(self):
        return len(self.boxes)

    @property
    def all(self) -> List[EvalBoxType]:
        """ Returns all EvalBoxes in a list. """
        ab = []
        for sample_token in self.sample_tokens:
            ab.extend(self[sample_token])
        return ab

    @property
    def sample_tokens(self) -> List[str]:
        """ Returns a list of all keys. """
        return list(self.boxes.keys())

    def add_boxes(self, sample_token: str, boxes: List[EvalBoxType]) -> None:
        """ Adds a list of boxes. """
        self.boxes[sample_token].extend(boxes)

    def serialize(self) -> dict:
        """ Serialize instance into json-friendly format. """
        return {key: [box.serialize() for box in boxes] for key, boxes in self.boxes.items()}

    @classmethod
    def deserialize(cls, content: dict, box_cls):
        """
        Initialize from serialized content.
        :param content: A dictionary with the serialized content of the box.
        :param box_cls: The class of the boxes, DetectionBox or TrackingBox.
        """
        eb = cls()
        for sample_token, boxes in content.items():
            eb.add_boxes(sample_token, [box_cls.deserialize(box) for box in boxes])
        return eb


class MetricData(abc.ABC):
    """ Abstract base class for the *MetricData classes specific to each task. """

    @abc.abstractmethod
    def serialize(self):
        """ Serialize instance into json-friendly format. """
        pass

    @classmethod
    @abc.abstractmethod
    def deserialize(cls, content: dict):
        """ Initialize from serialized content. """
        pass


class MOTAccumulatorCustom(motmetrics.mot.MOTAccumulator):
    def __init__(self):
        super().__init__()

    @staticmethod
    def new_event_dataframe_with_data(indices, events):
        """
        Create a new DataFrame filled with data.
        This version overwrites the original in MOTAccumulator achieves about 2x speedups.

        Params
        ------
        indices: list
            list of tuples (frameid, eventid)
        events: list
            list of events where each event is a list containing
            'Type', 'OId', HId', 'D'
        """
        idx = pd.MultiIndex.from_tuples(indices, names=['FrameId', 'Event'])
        df = pd.DataFrame(events, index=idx, columns=['Type', 'OId', 'HId', 'D'])
        return df

    @staticmethod
    def new_event_dataframe():
        """ Create a new DataFrame for event tracking. """
        idx = pd.MultiIndex(levels=[[], []], codes=[[], []], names=['FrameId', 'Event'])
        cats = pd.Categorical([], categories=['RAW', 'FP', 'MISS', 'SWITCH', 'MATCH'])
        df = pd.DataFrame(
            OrderedDict([
                ('Type', pd.Series(cats)),  # Type of event. One of FP (false positive), MISS, SWITCH, MATCH
                ('OId', pd.Series(dtype=object)),
                # Object ID or -1 if FP. Using float as missing values will be converted to NaN anyways.
                ('HId', pd.Series(dtype=object)),
                # Hypothesis ID or NaN if MISS. Using float as missing values will be converted to NaN anyways.
                ('D', pd.Series(dtype=float)),  # Distance or NaN when FP or MISS
            ]),
            index=idx
        )
        return df

    @property
    def events(self):
        if self.dirty_events:
            self.cached_events_df = MOTAccumulatorCustom.new_event_dataframe_with_data(self._indices, self._events)
            self.dirty_events = False
        return self.cached_events_df

    @staticmethod
    def merge_event_dataframes(dfs, update_frame_indices=True, update_oids=True, update_hids=True,
                               return_mappings=False):
        """Merge dataframes.

        Params
        ------
        dfs : list of pandas.DataFrame or MotAccumulator
            A list of event containers to merge

        Kwargs
        ------
        update_frame_indices : boolean, optional
            Ensure that frame indices are unique in the merged container
        update_oids : boolean, unique
            Ensure that object ids are unique in the merged container
        update_hids : boolean, unique
            Ensure that hypothesis ids are unique in the merged container
        return_mappings : boolean, unique
            Whether or not to return mapping information

        Returns
        -------
        df : pandas.DataFrame
            Merged event data frame
        """

        mapping_infos = []
        new_oid = count()
        new_hid = count()

        r = MOTAccumulatorCustom.new_event_dataframe()
        for df in dfs:

            if isinstance(df, MOTAccumulatorCustom):
                df = df.events

            copy = df.copy()
            infos = {}

            # Update index
            if update_frame_indices:
                next_frame_id = max(r.index.get_level_values(0).max() + 1,
                                    r.index.get_level_values(0).unique().shape[0])
                if np.isnan(next_frame_id):
                    next_frame_id = 0
                copy.index = copy.index.map(lambda x: (x[0] + next_frame_id, x[1]))
                infos['frame_offset'] = next_frame_id

            # Update object / hypothesis ids
            if update_oids:
                oid_map = dict([oid, str(next(new_oid))] for oid in copy['OId'].dropna().unique())
                copy['OId'] = copy['OId'].map(lambda x: oid_map[x], na_action='ignore')
                infos['oid_map'] = oid_map

            if update_hids:
                hid_map = dict([hid, str(next(new_hid))] for hid in copy['HId'].dropna().unique())
                copy['HId'] = copy['HId'].map(lambda x: hid_map[x], na_action='ignore')
                infos['hid_map'] = hid_map

            r = r.append(copy)
            mapping_infos.append(infos)

        if return_mappings:
            return r, mapping_infos
        else:
            return r

class TrackingMetricData(MetricData):
    """ This class holds accumulated and interpolated data required to calculate the tracking metrics. """

    nelem = None
    metrics = [m for m in list(set(TRACKING_METRICS) - set(AMOT_METRICS))]

    def __init__(self):
        # Set attributes explicitly to help IDEs figure out what is going on.
        assert TrackingMetricData.nelem is not None
        init = np.full(TrackingMetricData.nelem, np.nan)
        self.confidence = init
        self.recall_hypo = init
        self.recall = init
        self.motar = init
        self.mota = init
        self.motp = init
        self.faf = init
        self.gt = init
        self.tp = init
        self.mt = init
        self.ml = init
        self.fp = init
        self.fn = init
        self.ids = init
        self.frag = init
        self.tid = init
        self.lgd = init

    def __eq__(self, other):
        eq = True
        for key in self.serialize().keys():
            eq = eq and np.array_equal(getattr(self, key), getattr(other, key))
        return eq

    def __setattr__(self, *args, **kwargs):
        assert len(args) == 2
        name = args[0]
        values = np.array(args[1])
        assert values is None or len(values) == TrackingMetricData.nelem
        super(TrackingMetricData, self).__setattr__(name, values)

    def set_metric(self, metric_name: str, values: np.ndarray) -> None:
        """ Sets the specified metric. """
        self.__setattr__(metric_name, values)

    def get_metric(self, metric_name: str) -> np.ndarray:
        """ Returns the specified metric. """
        return self.__getattribute__(metric_name)

    @property
    def max_recall_ind(self):
        """ Returns index of max recall achieved. """

        # Last instance of confidence > 0 is index of max achieved recall.
        non_zero = np.nonzero(self.confidence)[0]
        if len(non_zero) == 0:  # If there are no matches, all the confidence values will be zero.
            max_recall_ind = 0
        else:
            max_recall_ind = non_zero[-1]

        return max_recall_ind

    @property
    def max_recall(self):
        """ Returns max recall achieved. """
        return self.recall[self.max_recall_ind]

    def serialize(self):
        """ Serialize instance into json-friendly format. """
        ret_dict = dict()
        for metric_name in ['confidence', 'recall_hypo'] + TrackingMetricData.metrics:
            ret_dict[metric_name] = self.get_metric(metric_name).tolist()
        return ret_dict

    @classmethod
    def set_nelem(cls, nelem: int) -> None:
        cls.nelem = nelem

    @classmethod
    def deserialize(cls, content: dict):
        """ Initialize from serialized content. """
        md = cls()
        for metric in ['confidence', 'recall_hypo'] + TrackingMetricData.metrics:
            md.set_metric(metric, content[metric])
        return md

    @classmethod
    def no_predictions(cls):
        """ Returns an md instance corresponding to having no predictions. """
        md = cls()
        md.confidence = np.zeros(cls.nelem)
        for metric in TrackingMetricData.metrics:
            md.set_metric(metric, np.zeros(cls.nelem))
        md.recall = np.linspace(0, 1, cls.nelem)
        return md

    @classmethod
    def random_md(cls):
        """ Returns an md instance corresponding to a random results. """
        md = cls()
        md.confidence = np.linspace(0, 1, cls.nelem)[::-1]
        for metric in TrackingMetricData.metrics:
            md.set_metric(metric, np.random.random(cls.nelem))
        md.recall = np.linspace(0, 1, cls.nelem)
        return md

class TrackingBox(EvalBox):
    """ Data class used during tracking evaluation. Can be a prediction or ground truth."""

    def __init__(self,
                 sample_token: str = "",
                 translation: Tuple[float, float, float] = (0, 0, 0),
                 size: Tuple[float, float, float] = (0, 0, 0),
                 rotation: Tuple[float, float, float, float] = (0, 0, 0, 0),
                 velocity: Tuple[float, float] = (0, 0),
                 ego_translation: Tuple[float, float, float] = (0, 0, 0),  # Translation to ego vehicle in meters.
                 num_pts: int = -1,  # Nbr. LIDAR or RADAR inside the box. Only for gt boxes.
                 tracking_id: str = '',  # Instance id of this object.
                 tracking_name: str = '',  # The class name used in the tracking challenge.
                 tracking_score: float = -1.0):  # Does not apply to GT.

        super().__init__(sample_token, translation, size, rotation, velocity, ego_translation, num_pts)

        assert tracking_name is not None, 'Error: tracking_name cannot be empty!'
        assert tracking_name in TRACKING_NAMES, 'Error: Unknown tracking_name %s' % tracking_name

        assert type(tracking_score) == float, 'Error: tracking_score must be a float!'
        assert not np.any(np.isnan(tracking_score)), 'Error: tracking_score may not be NaN!'

        # Assign.
        self.tracking_id = tracking_id
        self.tracking_name = tracking_name
        self.tracking_score = tracking_score

    def __eq__(self, other):
        return (self.sample_token == other.sample_token and
                self.translation == other.translation and
                self.size == other.size and
                self.rotation == other.rotation and
                self.velocity == other.velocity and
                self.ego_translation == other.ego_translation and
                self.num_pts == other.num_pts and
                self.tracking_id == other.tracking_id and
                self.tracking_name == other.tracking_name and
                self.tracking_score == other.tracking_score)

    def serialize(self) -> dict:
        """ Serialize instance into json-friendly format. """
        return {
            'sample_token': self.sample_token,
            'translation': self.translation,
            'size': self.size,
            'rotation': self.rotation,
            'velocity': self.velocity,
            'ego_translation': self.ego_translation,
            'num_pts': self.num_pts,
            'tracking_id': self.tracking_id,
            'tracking_name': self.tracking_name,
            'tracking_score': self.tracking_score
        }

    @classmethod
    def deserialize(cls, content: dict):
        """ Initialize from serialized content. """
        return cls(sample_token=content['sample_token'],
                   translation=tuple(content['translation']),
                   size=tuple(content['size']),
                   rotation=tuple(content['rotation']),
                   velocity=tuple(content['velocity']),
                   ego_translation=(0.0, 0.0, 0.0) if 'ego_translation' not in content
                   else tuple(content['ego_translation']),
                   num_pts=-1 if 'num_pts' not in content else int(content['num_pts']),
                   tracking_id=content['tracking_id'],
                   tracking_name=content['tracking_name'],
                   tracking_score=-1.0 if 'tracking_score' not in content else float(content['tracking_score']))

class TrackingMetricDataList:
    """ This stores a set of MetricData in a dict indexed by name. """

    def __init__(self):
        self.md: Dict[str, TrackingMetricData] = {}

    def __getitem__(self, key) -> TrackingMetricData:
        return self.md[key]

    def __eq__(self, other):
        eq = True
        for key in self.md.keys():
            eq = eq and self[key] == other[key]
        return eq

    def set(self, tracking_name: str, data: TrackingMetricData):
        """ Sets the MetricData entry for a certain tracking_name. """
        self.md[tracking_name] = data

    def serialize(self) -> dict:
        return {key: value.serialize() for key, value in self.md.items()}

    @classmethod
    def deserialize(cls, content: dict, metric_data_cls):
        mdl = cls()
        for name, md in content.items():
            mdl.set(name, metric_data_cls.deserialize(md))
        return mdl

def center_distance(gt_box: EvalBox, pred_box: EvalBox) -> float:
    """
    L2 distance between the box centers (xy only).
    :param gt_box: GT annotation sample.
    :param pred_box: Predicted sample.
    :return: L2 distance.
    """
    return np.linalg.norm(np.array(pred_box.translation[:2]) - np.array(gt_box.translation[:2]))


def velocity_l2(gt_box: EvalBox, pred_box: EvalBox) -> float:
    """
    L2 distance between the velocity vectors (xy only).
    If the predicted velocities are nan, we return inf, which is subsequently clipped to 1.
    :param gt_box: GT annotation sample.
    :param pred_box: Predicted sample.
    :return: L2 distance.
    """
    return np.linalg.norm(np.array(pred_box.velocity) - np.array(gt_box.velocity))


def yaw_diff(gt_box: EvalBox, eval_box: EvalBox, period: float = 2*np.pi) -> float:
    """
    Returns the yaw angle difference between the orientation of two boxes.
    :param gt_box: Ground truth box.
    :param eval_box: Predicted box.
    :param period: Periodicity in radians for assessing angle difference.
    :return: Yaw angle difference in radians in [0, pi].
    """
    yaw_gt = quaternion_yaw(Quaternion(gt_box.rotation))
    yaw_est = quaternion_yaw(Quaternion(eval_box.rotation))

    return abs(angle_diff(yaw_gt, yaw_est, period))


def angle_diff(x: float, y: float, period: float) -> float:
    """
    Get the smallest angle difference between 2 angles: the angle from y to x.
    :param x: To angle.
    :param y: From angle.
    :param period: Periodicity in radians for assessing angle difference.
    :return: <float>. Signed smallest between-angle difference in range (-pi, pi).
    """

    # calculate angle difference, modulo to [0, 2*pi]
    diff = (x - y + period / 2) % period - period / 2
    if diff > np.pi:
        diff = diff - (2 * np.pi)  # shift (pi, 2*pi] to (-pi, 0]

    return diff

class DetectionConfig:
    """ Data class that specifies the detection evaluation settings. """

    def __init__(self,
                 class_range: Dict[str, int],
                 dist_fcn: str,
                 dist_ths: List[float],
                 dist_th_tp: float,
                 min_recall: float,
                 min_precision: float,
                 max_boxes_per_sample: int,
                 mean_ap_weight: int):

        assert set(class_range.keys()) == set(DETECTION_NAMES), "Class count mismatch."
        assert dist_th_tp in dist_ths, "dist_th_tp must be in set of dist_ths."

        self.class_range = class_range
        self.dist_fcn = dist_fcn
        self.dist_ths = dist_ths
        self.dist_th_tp = dist_th_tp
        self.min_recall = min_recall
        self.min_precision = min_precision
        self.max_boxes_per_sample = max_boxes_per_sample
        self.mean_ap_weight = mean_ap_weight

        self.class_names = self.class_range.keys()

    def __eq__(self, other):
        eq = True
        for key in self.serialize().keys():
            eq = eq and np.array_equal(getattr(self, key), getattr(other, key))
        return eq

    def serialize(self) -> dict:
        """ Serialize instance into json-friendly format. """
        return {
            'class_range': self.class_range,
            'dist_fcn': self.dist_fcn,
            'dist_ths': self.dist_ths,
            'dist_th_tp': self.dist_th_tp,
            'min_recall': self.min_recall,
            'min_precision': self.min_precision,
            'max_boxes_per_sample': self.max_boxes_per_sample,
            'mean_ap_weight': self.mean_ap_weight
        }

    @classmethod
    def deserialize(cls, content: dict):
        """ Initialize from serialized dictionary. """
        return cls(content['class_range'],
                   content['dist_fcn'],
                   content['dist_ths'],
                   content['dist_th_tp'],
                   content['min_recall'],
                   content['min_precision'],
                   content['max_boxes_per_sample'],
                   content['mean_ap_weight'])

    @property
    def dist_fcn_callable(self):
        """ Return the distance function corresponding to the dist_fcn string. """
        if self.dist_fcn == 'center_distance':
            return center_distance
        else:
            raise Exception('Error: Unknown distance function %s!' % self.dist_fcn)


class DetectionMetricData(MetricData):
    """ This class holds accumulated and interpolated data required to calculate the detection metrics. """

    nelem = 101

    def __init__(self,
                 recall: np.array,
                 precision: np.array,
                 confidence: np.array,
                 trans_err: np.array,
                 vel_err: np.array,
                 scale_err: np.array,
                 orient_err: np.array,
                 attr_err: np.array):

        # Assert lengths.
        assert len(recall) == self.nelem
        assert len(precision) == self.nelem
        assert len(confidence) == self.nelem
        assert len(trans_err) == self.nelem
        assert len(vel_err) == self.nelem
        assert len(scale_err) == self.nelem
        assert len(orient_err) == self.nelem
        assert len(attr_err) == self.nelem

        # Assert ordering.
        assert all(confidence == sorted(confidence, reverse=True))  # Confidences should be descending.
        assert all(recall == sorted(recall))  # Recalls should be ascending.

        # Set attributes explicitly to help IDEs figure out what is going on.
        self.recall = recall
        self.precision = precision
        self.confidence = confidence
        self.trans_err = trans_err
        self.vel_err = vel_err
        self.scale_err = scale_err
        self.orient_err = orient_err
        self.attr_err = attr_err

    def __eq__(self, other):
        eq = True
        for key in self.serialize().keys():
            eq = eq and np.array_equal(getattr(self, key), getattr(other, key))
        return eq

    @property
    def max_recall_ind(self):
        """ Returns index of max recall achieved. """

        # Last instance of confidence > 0 is index of max achieved recall.
        non_zero = np.nonzero(self.confidence)[0]
        if len(non_zero) == 0:  # If there are no matches, all the confidence values will be zero.
            max_recall_ind = 0
        else:
            max_recall_ind = non_zero[-1]

        return max_recall_ind

    @property
    def max_recall(self):
        """ Returns max recall achieved. """

        return self.recall[self.max_recall_ind]

    def serialize(self):
        """ Serialize instance into json-friendly format. """
        return {
            'recall': self.recall.tolist(),
            'precision': self.precision.tolist(),
            'confidence': self.confidence.tolist(),
            'trans_err': self.trans_err.tolist(),
            'vel_err': self.vel_err.tolist(),
            'scale_err': self.scale_err.tolist(),
            'orient_err': self.orient_err.tolist(),
            'attr_err': self.attr_err.tolist(),
        }

    @classmethod
    def deserialize(cls, content: dict):
        """ Initialize from serialized content. """
        return cls(recall=np.array(content['recall']),
                   precision=np.array(content['precision']),
                   confidence=np.array(content['confidence']),
                   trans_err=np.array(content['trans_err']),
                   vel_err=np.array(content['vel_err']),
                   scale_err=np.array(content['scale_err']),
                   orient_err=np.array(content['orient_err']),
                   attr_err=np.array(content['attr_err']))

    @classmethod
    def no_predictions(cls):
        """ Returns a md instance corresponding to having no predictions. """
        return cls(recall=np.linspace(0, 1, cls.nelem),
                   precision=np.zeros(cls.nelem),
                   confidence=np.zeros(cls.nelem),
                   trans_err=np.ones(cls.nelem),
                   vel_err=np.ones(cls.nelem),
                   scale_err=np.ones(cls.nelem),
                   orient_err=np.ones(cls.nelem),
                   attr_err=np.ones(cls.nelem))

    @classmethod
    def random_md(cls):
        """ Returns an md instance corresponding to a random results. """
        return cls(recall=np.linspace(0, 1, cls.nelem),
                   precision=np.random.random(cls.nelem),
                   confidence=np.linspace(0, 1, cls.nelem)[::-1],
                   trans_err=np.random.random(cls.nelem),
                   vel_err=np.random.random(cls.nelem),
                   scale_err=np.random.random(cls.nelem),
                   orient_err=np.random.random(cls.nelem),
                   attr_err=np.random.random(cls.nelem))


class DetectionMetrics:
    """ Stores average precision and true positive metric results. Provides properties to summarize. """

    def __init__(self, cfg: DetectionConfig):

        self.cfg = cfg
        self._label_aps = defaultdict(lambda: defaultdict(float))
        self._label_tp_errors = defaultdict(lambda: defaultdict(float))
        self.eval_time = None

    def add_label_ap(self, detection_name: str, dist_th: float, ap: float) -> None:
        self._label_aps[detection_name][dist_th] = ap

    def get_label_ap(self, detection_name: str, dist_th: float) -> float:
        return self._label_aps[detection_name][dist_th]

    def add_label_tp(self, detection_name: str, metric_name: str, tp: float):
        self._label_tp_errors[detection_name][metric_name] = tp

    def get_label_tp(self, detection_name: str, metric_name: str) -> float:
        return self._label_tp_errors[detection_name][metric_name]

    def add_runtime(self, eval_time: float) -> None:
        self.eval_time = eval_time

    @property
    def mean_dist_aps(self) -> Dict[str, float]:
        """ Calculates the mean over distance thresholds for each label. """
        return {class_name: np.mean(list(d.values())) for class_name, d in self._label_aps.items()}

    @property
    def mean_ap(self) -> float:
        """ Calculates the mean AP by averaging over distance thresholds and classes. """
        return float(np.mean(list(self.mean_dist_aps.values())))

    @property
    def tp_errors(self) -> Dict[str, float]:
        """ Calculates the mean true positive error across all classes for each metric. """
        errors = {}
        for metric_name in TP_METRICS:
            class_errors = []
            for detection_name in self.cfg.class_names:
                class_errors.append(self.get_label_tp(detection_name, metric_name))

            errors[metric_name] = float(np.nanmean(class_errors))

        return errors

    @property
    def tp_scores(self) -> Dict[str, float]:
        scores = {}
        tp_errors = self.tp_errors
        for metric_name in TP_METRICS:

            # We convert the true positive errors to "scores" by 1-error.
            score = 1.0 - tp_errors[metric_name]

            # Some of the true positive errors are unbounded, so we bound the scores to min 0.
            score = max(0.0, score)

            scores[metric_name] = score

        return scores

    @property
    def nd_score(self) -> float:
        """
        Compute the nuScenes detection score (NDS, weighted sum of the individual scores).
        :return: The NDS.
        """
        # Summarize.
        total = float(self.cfg.mean_ap_weight * self.mean_ap + np.sum(list(self.tp_scores.values())))

        # Normalize.
        total = total / float(self.cfg.mean_ap_weight + len(self.tp_scores.keys()))

        return total

    def serialize(self):
        return {
            'label_aps': self._label_aps,
            'mean_dist_aps': self.mean_dist_aps,
            'mean_ap': self.mean_ap,
            'label_tp_errors': self._label_tp_errors,
            'tp_errors': self.tp_errors,
            'tp_scores': self.tp_scores,
            'nd_score': self.nd_score,
            'eval_time': self.eval_time,
            'cfg': self.cfg.serialize()
        }

    @classmethod
    def deserialize(cls, content: dict):
        """ Initialize from serialized dictionary. """

        cfg = DetectionConfig.deserialize(content['cfg'])

        metrics = cls(cfg=cfg)
        metrics.add_runtime(content['eval_time'])

        for detection_name, label_aps in content['label_aps'].items():
            for dist_th, ap in label_aps.items():
                metrics.add_label_ap(detection_name=detection_name, dist_th=float(dist_th), ap=float(ap))

        for detection_name, label_tps in content['label_tp_errors'].items():
            for metric_name, tp in label_tps.items():
                metrics.add_label_tp(detection_name=detection_name, metric_name=metric_name, tp=float(tp))

        return metrics

    def __eq__(self, other):
        eq = True
        eq = eq and self._label_aps == other._label_aps
        eq = eq and self._label_tp_errors == other._label_tp_errors
        eq = eq and self.eval_time == other.eval_time
        eq = eq and self.cfg == other.cfg

        return eq


class DetectionBox(EvalBox):
    """ Data class used during detection evaluation. Can be a prediction or ground truth."""

    def __init__(self,
                 sample_token: str = "",
                 translation: Tuple[float, float, float] = (0, 0, 0),
                 size: Tuple[float, float, float] = (0, 0, 0),
                 rotation: Tuple[float, float, float, float] = (0, 0, 0, 0),
                 velocity: Tuple[float, float] = (0, 0),
                 ego_translation: [float, float, float] = (0, 0, 0),  # Translation to ego vehicle in meters.
                 num_pts: int = -1,  # Nbr. LIDAR or RADAR inside the box. Only for gt boxes.
                 detection_name: str = 'car',  # The class name used in the detection challenge.
                 detection_score: float = -1.0,  # GT samples do not have a score.
                 attribute_name: str = ''):  # Box attribute. Each box can have at most 1 attribute.

        super().__init__(sample_token, translation, size, rotation, velocity, ego_translation, num_pts)

        assert detection_name is not None, 'Error: detection_name cannot be empty!'
        assert detection_name in DETECTION_NAMES, 'Error: Unknown detection_name %s' % detection_name

        assert attribute_name in ATTRIBUTE_NAMES or attribute_name == '', \
            'Error: Unknown attribute_name %s' % attribute_name

        assert type(detection_score) == float, 'Error: detection_score must be a float!'
        assert not np.any(np.isnan(detection_score)), 'Error: detection_score may not be NaN!'

        # Assign.
        self.detection_name = detection_name
        self.detection_score = detection_score
        self.attribute_name = attribute_name

    def __eq__(self, other):
        return (self.sample_token == other.sample_token and
                self.translation == other.translation and
                self.size == other.size and
                self.rotation == other.rotation and
                self.velocity == other.velocity and
                self.ego_translation == other.ego_translation and
                self.num_pts == other.num_pts and
                self.detection_name == other.detection_name and
                self.detection_score == other.detection_score and
                self.attribute_name == other.attribute_name)

    def serialize(self) -> dict:
        """ Serialize instance into json-friendly format. """
        return {
            'sample_token': self.sample_token,
            'translation': self.translation,
            'size': self.size,
            'rotation': self.rotation,
            'velocity': self.velocity,
            'ego_translation': self.ego_translation,
            'num_pts': self.num_pts,
            'detection_name': self.detection_name,
            'detection_score': self.detection_score,
            'attribute_name': self.attribute_name
        }

    @classmethod
    def deserialize(cls, content: dict):
        """ Initialize from serialized content. """
        return cls(sample_token=content['sample_token'],
                   translation=tuple(content['translation']),
                   size=tuple(content['size']),
                   rotation=tuple(content['rotation']),
                   velocity=tuple(content['velocity']),
                   ego_translation=(0.0, 0.0, 0.0) if 'ego_translation' not in content
                   else tuple(content['ego_translation']),
                   num_pts=-1 if 'num_pts' not in content else int(content['num_pts']),
                   detection_name=content['detection_name'],
                   detection_score=-1.0 if 'detection_score' not in content else float(content['detection_score']),
                   attribute_name=content['attribute_name'])


class DetectionMetricDataList:
    """ This stores a set of MetricData in a dict indexed by (name, match-distance). """

    def __init__(self):
        self.md = {}

    def __getitem__(self, key):
        return self.md[key]

    def __eq__(self, other):
        eq = True
        for key in self.md.keys():
            eq = eq and self[key] == other[key]
        return eq

    def get_class_data(self, detection_name: str) -> List[Tuple[DetectionMetricData, float]]:
        """ Get all the MetricData entries for a certain detection_name. """
        return [(md, dist_th) for (name, dist_th), md in self.md.items() if name == detection_name]

    def get_dist_data(self, dist_th: float) -> List[Tuple[DetectionMetricData, str]]:
        """ Get all the MetricData entries for a certain match_distance. """
        return [(md, detection_name) for (detection_name, dist), md in self.md.items() if dist == dist_th]

    def set(self, detection_name: str, match_distance: float, data: DetectionMetricData):
        """ Sets the MetricData entry for a certain detection_name and match_distance. """
        self.md[(detection_name, match_distance)] = data

    def serialize(self) -> dict:
        return {key[0] + ':' + str(key[1]): value.serialize() for key, value in self.md.items()}

    @classmethod
    def deserialize(cls, content: dict):
        mdl = cls()
        for key, md in content.items():
            name, distance = key.split(':')
            mdl.set(name, float(distance), DetectionMetricData.deserialize(md))
        return mdl

def attr_acc(gt_box: DetectionBox, pred_box: DetectionBox) -> float:
    """
    Computes the classification accuracy for the attribute of this class (if any).
    If the GT class has no attributes or the annotation is missing attributes, we assign an accuracy of nan, which is
    ignored later on.
    :param gt_box: GT annotation sample.
    :param pred_box: Predicted sample.
    :return: Attribute classification accuracy (0 or 1) or nan if GT annotation does not have any attributes.
    """
    if gt_box.attribute_name == '':
        # If the class does not have attributes or this particular sample is missing attributes, return nan, which is
        # ignored later. Note that about 0.4% of the sample_annotations have no attributes, although they should.
        acc = np.nan
    else:
        # Check that label is correct.
        acc = float(gt_box.attribute_name == pred_box.attribute_name)
    return acc


def scale_iou(sample_annotation: EvalBox, sample_result: EvalBox) -> float:
    """
    This method compares predictions to the ground truth in terms of scale.
    It is equivalent to intersection over union (IOU) between the two boxes in 3D,
    if we assume that the boxes are aligned, i.e. translation and rotation are considered identical.
    :param sample_annotation: GT annotation sample.
    :param sample_result: Predicted sample.
    :return: Scale IOU.
    """
    # Validate inputs.
    sa_size = np.array(sample_annotation.size)
    sr_size = np.array(sample_result.size)
    assert all(sa_size > 0), 'Error: sample_annotation sizes must be >0.'
    assert all(sr_size > 0), 'Error: sample_result sizes must be >0.'

    # Compute IOU.
    min_wlh = np.minimum(sa_size, sr_size)
    volume_annotation = np.prod(sa_size)
    volume_result = np.prod(sr_size)
    intersection = np.prod(min_wlh)  # type: float
    union = volume_annotation + volume_result - intersection  # type: float
    iou = intersection / union

    return iou


def quaternion_yaw(q: Quaternion) -> float:
    """
    Calculate the yaw angle from a quaternion.
    Note that this only works for a quaternion that represents a box in lidar or global coordinate frame.
    It does not work for a box in the camera frame.
    :param q: Quaternion of interest.
    :return: Yaw angle in radians.
    """

    # Project into xy plane.
    v = np.dot(q.rotation_matrix, np.array([1, 0, 0]))

    # Measure yaw using arctan.
    yaw = np.arctan2(v[1], v[0])

    return yaw


def boxes_to_sensor(boxes: List[EvalBox], pose_record: Dict, cs_record: Dict):
    """
    Map boxes from global coordinates to the vehicle's sensor coordinate system.
    :param boxes: The boxes in global coordinates.
    :param pose_record: The pose record of the vehicle at the current timestamp.
    :param cs_record: The calibrated sensor record of the sensor.
    :return: The transformed boxes.
    """
    boxes_out = []
    for box in boxes:
        # Create Box instance.
        box = Box(box.translation, box.size, Quaternion(box.rotation))

        # Move box to ego vehicle coord system.
        box.translate(-np.array(pose_record['translation']))
        box.rotate(Quaternion(pose_record['rotation']).inverse)

        #  Move box to sensor coord system.
        box.translate(-np.array(cs_record['translation']))
        box.rotate(Quaternion(cs_record['rotation']).inverse)

        boxes_out.append(box)

    return boxes_out


def cummean(x: np.array) -> np.array:
    """
    Computes the cumulative mean up to each position in a NaN sensitive way
    - If all values are NaN return an array of ones.
    - If some values are NaN, accumulate arrays discording those entries.
    """
    if sum(np.isnan(x)) == len(x):
        # Is all numbers in array are NaN's.
        return np.ones(len(x))  # If all errors are NaN set to error to 1 for all operating points.
    else:
        # Accumulate in a nan-aware manner.
        sum_vals = np.nancumsum(x.astype(float))  # Cumulative sum ignoring nans.
        count_vals = np.cumsum(~np.isnan(x))  # Number of non-nans up to each position.
        return np.divide(sum_vals, count_vals, out=np.zeros_like(sum_vals), where=count_vals != 0)

class TrackingConfig:
    """ Data class that specifies the tracking evaluation settings. """

    def __init__(self,
                 tracking_names: List[str],
                 pretty_tracking_names: Dict[str, str],
                 tracking_colors: Dict[str, str],
                 class_range: Dict[str, int],
                 dist_fcn: str,
                 dist_th_tp: float,
                 min_recall: float,
                 max_boxes_per_sample: float,
                 metric_worst: Dict[str, float],
                 num_thresholds: int):

        assert set(class_range.keys()) == set(tracking_names), "Class count mismatch."
        global TRACKING_NAMES
        TRACKING_NAMES = tracking_names
        self.tracking_names = tracking_names
        self.pretty_tracking_names = pretty_tracking_names
        self.tracking_colors = tracking_colors
        self.class_range = class_range
        self.dist_fcn = dist_fcn
        self.dist_th_tp = dist_th_tp
        self.min_recall = min_recall
        self.max_boxes_per_sample = max_boxes_per_sample
        self.metric_worst = metric_worst
        self.num_thresholds = num_thresholds

        TrackingMetricData.set_nelem(num_thresholds)

        self.class_names = sorted(self.class_range.keys())

    def __eq__(self, other):
        eq = True
        for key in self.serialize().keys():
            eq = eq and np.array_equal(getattr(self, key), getattr(other, key))
        return eq

    def serialize(self) -> dict:
        """ Serialize instance into json-friendly format. """
        return {
            'tracking_names': self.tracking_names,
            'pretty_tracking_names': self.pretty_tracking_names,
            'tracking_colors': self.tracking_colors,
            'class_range': self.class_range,
            'dist_fcn': self.dist_fcn,
            'dist_th_tp': self.dist_th_tp,
            'min_recall': self.min_recall,
            'max_boxes_per_sample': self.max_boxes_per_sample,
            'metric_worst': self.metric_worst,
            'num_thresholds': self.num_thresholds
        }

    @classmethod
    def deserialize(cls, content: dict):
        """ Initialize from serialized dictionary. """
        return cls(content['tracking_names'],
                   content['pretty_tracking_names'],
                   content['tracking_colors'],
                   content['class_range'],
                   content['dist_fcn'],
                   content['dist_th_tp'],
                   content['min_recall'],
                   content['max_boxes_per_sample'],
                   content['metric_worst'],
                   content['num_thresholds'])

    @property
    def dist_fcn_callable(self):
        """ Return the distance function corresponding to the dist_fcn string. """
        if self.dist_fcn == 'center_distance':
            return center_distance
        else:
            raise Exception('Error: Unknown distance function %s!' % self.dist_fcn)
class TrackingMetrics:
    """ Stores tracking metric results. Provides properties to summarize. """

    def __init__(self, cfg: TrackingConfig):

        self.cfg = cfg
        self.eval_time = None
        self.label_metrics: Dict[str, Dict[str, float]] = {}
        self.class_names = self.cfg.class_names
        self.metric_names = [l for l in TRACKING_METRICS]

        # Init every class.
        for metric_name in self.metric_names:
            self.label_metrics[metric_name] = {}
            for class_name in self.class_names:
                self.label_metrics[metric_name][class_name] = np.nan

    def add_label_metric(self, metric_name: str, tracking_name: str, value: float) -> None:
        assert metric_name in self.label_metrics
        self.label_metrics[metric_name][tracking_name] = float(value)

    def add_runtime(self, eval_time: float) -> None:
        self.eval_time = eval_time

    def compute_metric(self, metric_name: str, class_name: str = 'all') -> float:
        if class_name == 'all':
            data = list(self.label_metrics[metric_name].values())
            if len(data) > 0:
                # Some metrics need to be summed, not averaged.
                # Nan entries are ignored.
                if metric_name in ['mt', 'ml', 'tp', 'fp', 'fn', 'ids', 'frag']:
                    return float(np.nansum(data))
                else:
                    return float(np.nanmean(data))
            else:
                return np.nan
        else:
            return float(self.label_metrics[metric_name][class_name])

    def serialize(self) -> Dict[str, Any]:
        metrics = dict()
        metrics['label_metrics'] = self.label_metrics
        metrics['eval_time'] = self.eval_time
        metrics['cfg'] = self.cfg.serialize()
        for metric_name in self.label_metrics.keys():
            metrics[metric_name] = self.compute_metric(metric_name)

        return metrics

    @classmethod
    def deserialize(cls, content: dict) -> 'TrackingMetrics':
        """ Initialize from serialized dictionary. """
        cfg = TrackingConfig.deserialize(content['cfg'])
        tm = cls(cfg=cfg)
        tm.add_runtime(content['eval_time'])
        tm.label_metrics = content['label_metrics']

        return tm

    def __eq__(self, other):
        eq = True
        eq = eq and self.label_metrics == other.label_metrics
        eq = eq and self.eval_time == other.eval_time
        eq = eq and self.cfg == other.cfg

        return eq


def config_factory(configuration_name: str) -> Union[DetectionConfig, TrackingConfig]:
    """
    Creates a *Config instance that can be used to initialize a *Eval instance, where * stands for Detection/Tracking.
    Note that this only works if the config file is located in the nuscenes/eval/common/configs folder.
    :param configuration_name: Name of desired configuration in eval_detection_configs.
    :return: *Config instance.
    """
    # Check if config exists.
    tokens = configuration_name.split('_')
    assert len(tokens) > 1, 'Error: Configuration name must be have prefix "detection_" or "tracking_"!'
    task = tokens[0]
    cfg_path = configuration_name
    assert os.path.exists(cfg_path), 'Requested unknown configuration {}'.format(configuration_name)

    # Load config file and deserialize it.
    with open(cfg_path, 'r') as f:
        data = json.load(f)
    cfg = TrackingConfig.deserialize(data)

    return cfg

class TrackingEvaluation(object):
    def __init__(self,
                 nusc, 
                 tracks_gt: Dict[str, Dict[int, List[TrackingBox]]],
                 tracks_pred: Dict[str, Dict[int, List[TrackingBox]]],
                 class_name: str,
                 dist_fcn: Callable,
                 dist_th_tp: float,
                 min_recall: float,
                 num_thresholds: int,
                 metric_worst: Dict[str, float],
                 verbose: bool = True,
                 output_dir: str = None,
                 render_classes: List[str] = None):
        """
        Create a TrackingEvaluation object which computes all metrics for a given class.
        :param tracks_gt: The ground-truth tracks.
        :param tracks_pred: The predicted tracks.
        :param class_name: The current class we are evaluating on.
        :param dist_fcn: The distance function used for evaluation.
        :param dist_th_tp: The distance threshold used to determine matches.
        :param min_recall: The minimum recall value below which we drop thresholds due to too much noise.
        :param num_thresholds: The number of recall thresholds from 0 to 1. Note that some of these may be dropped.
        :param metric_worst: Mapping from metric name to the fallback value assigned if a recall threshold
            is not achieved.
        :param verbose: Whether to print to stdout.
        :param output_dir: Output directory to save renders.
        :param render_classes: Classes to render to disk or None.

        Computes the metrics defined in:
        - Stiefelhagen 2008: Evaluating Multiple Object Tracking Performance: The CLEAR MOT Metrics.
          MOTA, MOTP
        - Nevatia 2008: Global Data Association for Multi-Object Tracking Using Network Flows.
          MT/PT/ML
        - Weng 2019: "A Baseline for 3D Multi-Object Tracking".
          AMOTA/AMOTP
        """
        self.nusc=nusc
        self.tracks_gt = tracks_gt
        self.tracks_pred = tracks_pred
        self.class_name = class_name
        self.dist_fcn = dist_fcn
        self.dist_th_tp = dist_th_tp
        self.min_recall = min_recall
        self.num_thresholds = num_thresholds
        self.metric_worst = metric_worst
        self.verbose = verbose
        self.output_dir = output_dir
        self.render_classes = [] if render_classes is None else render_classes

        self.n_scenes = len(self.tracks_gt)

        # Specify threshold naming pattern. Note that no two thresholds may have the same name.
        def name_gen(_threshold):
            return 'thr_%.4f' % _threshold
        self.name_gen = name_gen

        # Check that metric definitions are consistent.
        for metric_name in MOT_METRIC_MAP.values():
            assert metric_name == '' or metric_name in TRACKING_METRICS

    def accumulate(self) -> TrackingMetricData:
        """
        Compute metrics for all recall thresholds of the current class.
        :return: TrackingMetricData instance which holds the metrics for each threshold.
        """
        # Init.
        print('Computing metrics for class %s...\n' % self.class_name)
        accumulators = []
        thresh_metrics = []
        md = TrackingMetricData()

        # Skip missing classes.
        gt_box_count = 0
        gt_track_ids = set()
        for scene_tracks_gt in self.tracks_gt.values():
            for frame_gt in scene_tracks_gt.values():
                for box in frame_gt:
                    if box.tracking_name == self.class_name:
                        gt_box_count += 1
                        gt_track_ids.add(box.tracking_id)
        if gt_box_count == 0:
            # Do not add any metric. The average metrics will then be nan.
            return md

        # Register mot metrics.
        mh = create_motmetrics()

        # Get thresholds.
        # Note: The recall values are the hypothetical recall (10%, 20%, ..).
        # The actual recall may vary as there is no way to compute it without trying all thresholds.
        thresholds, recalls = self.compute_thresholds(gt_box_count)
        md.confidence = thresholds
        md.recall_hypo = recalls
        print('Computed thresholds\n')

        for t, threshold in enumerate(thresholds):
            # If recall threshold is not achieved, we assign the worst possible value in AMOTA and AMOTP.
            if np.isnan(threshold):
                continue

            # Do not compute the same threshold twice.
            # This becomes relevant when a user submits many boxes with the exact same score.
            if threshold in thresholds[:t]:
                continue

            # Accumulate track data.
            acc, _ = self.accumulate_threshold(threshold)
            accumulators.append(acc)

            # Compute metrics for current threshold.
            thresh_name = self.name_gen(threshold)
            thresh_summary = mh.compute(
                acc, metrics=MOT_METRIC_MAP.keys(), name=thresh_name)
            thresh_metrics.append(thresh_summary)

            # Print metrics to stdout.
            print_threshold_metrics(thresh_summary.to_dict())

        # Concatenate all metrics. We only do this for more convenient access.
        if len(thresh_metrics) == 0:
            summary = []
        else:
            summary = pandas.concat(thresh_metrics)

        # Sanity checks.
        unachieved_thresholds = np.sum(np.isnan(thresholds))
        duplicate_thresholds = len(thresholds) - len(np.unique(thresholds))
        #assert unachieved_thresholds + duplicate_thresholds + len(thresh_metrics) == self.num_thresholds

        # Figure out how many times each threshold should be repeated.
        valid_thresholds = [t for t in thresholds if not np.isnan(t)]
        assert valid_thresholds == sorted(valid_thresholds)
        rep_counts = [np.sum(thresholds == t)
                      for t in np.unique(valid_thresholds)]

        # Store all traditional metrics.
        for (mot_name, metric_name) in MOT_METRIC_MAP.items():
            # Skip metrics which we don't output.
            if metric_name == '':
                continue

            # Retrieve and store values for current metric.
            if len(thresh_metrics) == 0:
                # Set all the worst possible value if no recall threshold is achieved.
                worst = self.metric_worst[metric_name]
                if worst == -1:
                    if metric_name == 'ml':
                        worst = len(gt_track_ids)
                    elif metric_name in ['gt', 'fn']:
                        worst = gt_box_count
                    elif metric_name in ['fp', 'ids', 'frag']:
                        # We can't know how these error types are distributed.
                        worst = np.nan
                    else:
                        raise NotImplementedError

                all_values = [worst] * TrackingMetricData.nelem
            else:
                values = summary.get(mot_name).values
                assert np.all(values[np.logical_not(np.isnan(values))] >= 0)

                # If a threshold occurred more than once, duplicate the metric values.
                assert len(rep_counts) == len(values)
                values = np.concatenate([([v] * r)
                                        for (v, r) in zip(values, rep_counts)])

                # Pad values with nans for unachieved recall thresholds.
                all_values = [np.nan] * unachieved_thresholds
                all_values.extend(values)

            assert len(all_values) == TrackingMetricData.nelem
            md.set_metric(metric_name, all_values)

        return md

    def accumulate_threshold(self, threshold: float = None) -> Tuple[pandas.DataFrame, List[float]]:
        """
        Accumulate metrics for a particular recall threshold of the current class.
        The scores are only computed if threshold is set to None. This is used to infer the recall thresholds.
        :param threshold: score threshold used to determine positives and negatives.
        :return: (The MOTAccumulator that stores all the hits/misses/etc, Scores for each TP).
        """
        accs = []
        # The scores of the TPs. These are used to determine the recall thresholds initially.
        scores = []

        # Go through all frames and associate ground truth and tracker results.
        # Groundtruth and tracker contain lists for every single frame containing lists detections.
        for scene_id in self.tracks_gt.keys():

            # Initialize accumulator and frame_id for this scene
            acc = MOTAccumulatorCustom()
            frame_id = 0  # Frame ids must be unique across all scenes

            # Retrieve GT and preds.
            scene_tracks_gt = self.tracks_gt[scene_id]
            scene_tracks_pred = self.tracks_pred[scene_id]

            # Visualize the boxes in this frame.
            if self.class_name in self.render_classes and threshold is None:
                save_path = os.path.join(
                    self.output_dir, 'render', str(scene_id), self.class_name)
                os.makedirs(save_path, exist_ok=True)
                renderer = TrackingRenderer(scene_id, save_path)
            else:
                renderer = None

            for timestamp in scene_tracks_gt.keys():
                # Select only the current class.
                frame_gt = scene_tracks_gt[timestamp]
                frame_pred = scene_tracks_pred[timestamp]
                frame_gt = [
                    f for f in frame_gt if f.tracking_name == self.class_name]
                frame_pred = [
                    f for f in frame_pred if f.tracking_name == self.class_name]

                # Threshold boxes by score. Note that the scores were previously averaged over the whole track.
                if threshold is not None:
                    frame_pred = [
                        f for f in frame_pred if f.tracking_score >= threshold]

                # Abort if there are neither GT nor pred boxes.
                gt_ids = [gg.tracking_id for gg in frame_gt]
                pred_ids = [tt.tracking_id for tt in frame_pred]
                if len(gt_ids) == 0 and len(pred_ids) == 0:
                    continue

                # Calculate distances.
                # Note that the distance function is hard-coded to achieve significant speedups via vectorization.
                assert self.dist_fcn.__name__ == 'center_distance'
                if len(frame_gt) == 0 or len(frame_pred) == 0:
                    distances = np.ones((0, 0))
                else:
                    gt_boxes = np.array([b.translation[:2] for b in frame_gt])
                    pred_boxes = np.array([b.translation[:2]
                                          for b in frame_pred])
                    distances = sklearn.metrics.pairwise.euclidean_distances(
                        gt_boxes, pred_boxes)

                # Distances that are larger than the threshold won't be associated.
                assert len(distances) == 0 or not np.all(np.isnan(distances))
                distances[distances >= self.dist_th_tp] = np.nan

                # Accumulate results.
                # Note that we cannot use timestamp as frameid as motmetrics assumes it's an integer.
                acc.update(gt_ids, pred_ids, distances, frameid=frame_id)

                # Store scores of matches, which are used to determine recall thresholds.
                if threshold is None:
                    events = acc.events.loc[frame_id]
                    matches = events[events.Type == 'MATCH']
                    match_ids = matches.HId.values
                    match_scores = [
                        tt.tracking_score for tt in frame_pred if tt.tracking_id in match_ids]
                    scores.extend(match_scores)
                else:
                    events = None


                # Render the boxes in this frame.
                if self.class_name in self.render_classes and threshold is None:
                    renderer.render(self.nusc, events, timestamp, frame_gt, frame_pred)
                #if self.class_name in self.render_classes:
                #    renderer.render(events, timestamp, frame_gt, frame_pred)

                # Increment the frame_id, unless there are no boxes (equivalent to what motmetrics does).
                frame_id += 1

            accs.append(acc)

        # Merge accumulators
        acc_merged = MOTAccumulatorCustom.merge_event_dataframes(accs)
        return acc_merged, scores

    def compute_thresholds(self, gt_box_count: int) -> Tuple[List[float], List[float]]:
        """
        Compute the score thresholds for predefined recall values.
        AMOTA/AMOTP average over all thresholds, whereas MOTA/MOTP/.. pick the threshold with the highest MOTA.
        :param gt_box_count: The number of GT boxes for this class.
        :return: The lists of thresholds and their recall values.
        """
        # Run accumulate to get the scores of TPs.
        _, scores = self.accumulate_threshold(threshold=None)

        # Abort if no predictions exist.
        if len(scores) == 0:
            return [np.nan] * self.num_thresholds, [np.nan] * self.num_thresholds

        # Sort scores.
        scores = np.array(scores)
        scores.sort()
        scores = scores[::-1]

        # Compute recall levels.
        tps = np.array(range(1, len(scores) + 1))
        rec = tps / gt_box_count
        #assert len(scores) / gt_box_count <= 1

        # Determine thresholds.
        max_recall_achieved = np.max(rec)
        rec_interp = np.linspace(
            self.min_recall, 1, self.num_thresholds).round(12)
        thresholds = np.interp(rec_interp, rec, scores, right=0)

        # Set thresholds for unachieved recall values to nan to penalize AMOTA/AMOTP later.
        thresholds[rec_interp > max_recall_achieved] = np.nan

        # Cast to list.
        thresholds = list(thresholds.tolist())
        rec_interp = list(rec_interp.tolist())

        # Reverse order for more convenient presentation.
        thresholds.reverse()
        rec_interp.reverse()

        # Check that we return the correct number of thresholds.
        assert len(thresholds) == len(rec_interp) == self.num_thresholds

        return thresholds, rec_interp

def summary_plot(cfg: TrackingConfig,
                 md_list: TrackingMetricDataList,
                 ncols: int = 2,
                 savepath: str = None) -> None:
    """
    Creates a summary plot with which includes all traditional metrics for each class.
    :param cfg: A TrackingConfig object.
    :param md_list: TrackingMetricDataList instance.
    :param ncols: How many columns the resulting plot should have.
    :param savepath: If given, saves the the rendering here instead of displaying.
    """
    # Select metrics and setup plot.
    rel_metrics = ['motar', 'motp', 'mota', 'recall', 'mt', 'ml', 'faf', 'tp', 'fp', 'fn', 'ids', 'frag', 'tid', 'lgd']
    n_metrics = len(rel_metrics)
    nrows = int(np.ceil(n_metrics / ncols))
    _, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7.5 * ncols, 5 * nrows))

    # For each metric, plot all the classes in one diagram.
    for ind, metric_name in enumerate(rel_metrics):
        row = ind // ncols
        col = np.mod(ind, ncols)
        recall_metric_curve(cfg, md_list, metric_name, ax=axes[row, col])

    # Set layout with little white space and save to disk.
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath)
        plt.close()
def setup_axis(xlabel: str = None,
               ylabel: str = None,
               xlim: int = None,
               ylim: int = None,
               title: str = None,
               min_precision: float = None,
               min_recall: float = None,
               ax: Axis = None,
               show_spines: str = 'none'):
    """
    Helper method that sets up the axis for a plot.
    :param xlabel: x label text.
    :param ylabel: y label text.
    :param xlim: Upper limit for x axis.
    :param ylim: Upper limit for y axis.
    :param title: Axis title.
    :param min_precision: Visualize minimum precision as horizontal line.
    :param min_recall: Visualize minimum recall as vertical line.
    :param ax: (optional) an existing axis to be modified.
    :param show_spines: Whether to show axes spines, set to 'none' by default.
    :return: The axes object.
    """
    if ax is None:
        ax = plt.subplot()

    ax.get_xaxis().tick_bottom()
    ax.tick_params(labelsize=16)
    ax.get_yaxis().tick_left()

    # Hide the selected axes spines.
    if show_spines in ['bottomleft', 'none']:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if show_spines == 'none':
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
    elif show_spines in ['all']:
        pass
    else:
        raise NotImplementedError

    if title is not None:
        ax.set_title(title, size=24)
    if xlabel is not None:
        ax.set_xlabel(xlabel, size=16)
    if ylabel is not None:
        ax.set_ylabel(ylabel, size=16)
    if xlim is not None:
        ax.set_xlim(0, xlim)
    if ylim is not None:
        ax.set_ylim(0, ylim)
    if min_recall is not None:
        ax.axvline(x=min_recall, linestyle='--', color=(0, 0, 0, 0.3))
    if min_precision is not None:
        ax.axhline(y=min_precision, linestyle='--', color=(0, 0, 0, 0.3))

    return ax

def recall_metric_curve(cfg: TrackingConfig,
                        md_list: TrackingMetricDataList,
                        metric_name: str,
                        savepath: str = None,
                        ax: Axis = None) -> None:
    """
    Plot the recall versus metric curve for the given metric.
    :param cfg: A TrackingConfig object.
    :param md_list: TrackingMetricDataList instance.
    :param metric_name: The name of the metric to plot.
    :param savepath: If given, saves the the rendering here instead of displaying.
    :param ax: Axes onto which to render or None to create a new axis.
    """
    min_recall = cfg.min_recall  # Minimum recall value from config.
    # Setup plot.
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(7.5, 5))
    ax = setup_axis(xlabel='Recall', ylabel=metric_name.upper(),
                    xlim=1, ylim=None, min_recall=min_recall, ax=ax, show_spines='bottomleft')

    # Plot the recall vs. precision curve for each detection class.
    for tracking_name, md in md_list.md.items():
        # Get values.
        confidence = md.confidence
        recalls = md.recall_hypo
        values = md.get_metric(metric_name)

        # Filter unachieved recall thresholds.
        valid = np.where(np.logical_not(np.isnan(confidence)))[0]
        if len(valid) == 0:
            continue
        first_valid = valid[0]
        assert not np.isnan(confidence[-1])
        recalls = recalls[first_valid:]
        values = values[first_valid:]

        ax.plot(recalls,
                values,
                label='%s' % cfg.pretty_tracking_names[tracking_name],
                color=cfg.tracking_colors[tracking_name])

    # Scale count statistics and FAF logarithmically.
    if metric_name in ['mt', 'ml', 'faf', 'tp', 'fp', 'fn', 'ids', 'frag']:
        ax.set_yscale('symlog')

    if metric_name in ['amota', 'motar', 'recall', 'mota']:
        # Some metrics have an upper bound of 1.
        ax.set_ylim(0, 1)
    elif metric_name != 'motp':
        # For all other metrics except MOTP we set a lower bound of 0.
        ax.set_ylim(bottom=0)

    ax.legend(loc='upper right', borderaxespad=0)
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath)
        plt.close()


class PointCloud(ABC):
    """
    Abstract class for manipulating and viewing point clouds.
    Every point cloud (lidar and radar) consists of points where:
    - Dimensions 0, 1, 2 represent x, y, z coordinates.
        These are modified when the point cloud is rotated or translated.
    - All other dimensions are optional. Hence these have to be manually modified if the reference frame changes.
    """

    def __init__(self, points: np.ndarray):
        """
        Initialize a point cloud and check it has the correct dimensions.
        :param points: <np.float: d, n>. d-dimensional input point cloud matrix.
        """
        assert points.shape[0] == self.nbr_dims(), 'Error: Pointcloud points must have format: %d x n' % self.nbr_dims()
        self.points = points

    @staticmethod
    @abstractmethod
    def nbr_dims() -> int:
        """
        Returns the number of dimensions.
        :return: Number of dimensions.
        """
        pass

    @classmethod
    @abstractmethod
    def from_file(cls, file_name: str) -> 'PointCloud':
        """
        Loads point cloud from disk.
        :param file_name: Path of the pointcloud file on disk.
        :return: PointCloud instance.
        """
        pass

    @classmethod
    def from_file_multisweep(cls,
                             nusc: 'NuScenes',
                             sample_rec: Dict,
                             chan: str,
                             ref_chan: str,
                             nsweeps: int = 1,
                             min_distance: float = 1.0) -> Tuple['PointCloud', np.ndarray]:
        """
        Return a point cloud that aggregates multiple sweeps.
        As every sweep is in a different coordinate frame, we need to map the coordinates to a single reference frame.
        As every sweep has a different timestamp, we need to account for that in the transformations and timestamps.
        :param nusc: A NuScenes instance.
        :param sample_rec: The current sample.
        :param chan: The lidar/radar channel from which we track back n sweeps to aggregate the point cloud.
        :param ref_chan: The reference channel of the current sample_rec that the point clouds are mapped to.
        :param nsweeps: Number of sweeps to aggregated.
        :param min_distance: Distance below which points are discarded.
        :return: (all_pc, all_times). The aggregated point cloud and timestamps.
        """
        # Init.
        points = np.zeros((cls.nbr_dims(), 0), dtype=np.float32 if cls == LidarPointCloud else np.float64)
        all_pc = cls(points)
        all_times = np.zeros((1, 0))

        # Get reference pose and timestamp.
        ref_sd_token = sample_rec['data'][ref_chan]
        ref_sd_rec = nusc.get('sample_data', ref_sd_token)
        ref_pose_rec = nusc.get('ego_pose', ref_sd_rec['ego_pose_token'])
        ref_cs_rec = nusc.get('calibrated_sensor', ref_sd_rec['calibrated_sensor_token'])
        ref_time = 1e-6 * ref_sd_rec['timestamp']

        # Homogeneous transform from ego car frame to reference frame.
        ref_from_car = transform_matrix(ref_cs_rec['translation'], Quaternion(ref_cs_rec['rotation']), inverse=True)

        # Homogeneous transformation matrix from global to _current_ ego car frame.
        car_from_global = transform_matrix(ref_pose_rec['translation'], Quaternion(ref_pose_rec['rotation']),
                                           inverse=True)

        # Aggregate current and previous sweeps.
        sample_data_token = sample_rec['data'][chan]
        current_sd_rec = nusc.get('sample_data', sample_data_token)
        
        # Load up the pointcloud and remove points close to the sensor.
        current_pc = cls.from_file(osp.join(nusc.dataroot, current_sd_rec['filename']))
        current_pc.remove_close(min_distance)
            
        # Get past pose.
        current_pose_rec = nusc.get('ego_pose', current_sd_rec['ego_pose_token'])
        global_from_car = transform_matrix(current_pose_rec['translation'],
                                               Quaternion(current_pose_rec['rotation']), inverse=False)

        # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
        current_cs_rec = nusc.get('calibrated_sensor', current_sd_rec['calibrated_sensor_token'])
        car_from_current = transform_matrix(current_cs_rec['translation'], Quaternion(current_cs_rec['rotation']),
                                                inverse=False)

        # Fuse four transformation matrices into one and perform transform.
        #trans_matrix = reduce(np.dot, [ref_from_car, car_from_global, global_from_car, car_from_current])
        trans_matrix = reduce(np.dot, [global_from_car,car_from_current])
        current_pc.transform(trans_matrix)
        current_pc.points[0,:]=current_pc.points[0,:]-ref_pose_rec['translation'][0]
        current_pc.points[1,:]=current_pc.points[1,:]-ref_pose_rec['translation'][1]
        current_pc.points[2,:]=current_pc.points[2,:]-ref_pose_rec['translation'][2]

    
        return current_pc

    def nbr_points(self) -> int:
        """
        Returns the number of points.
        :return: Number of points.
        """
        return self.points.shape[1]

    def subsample(self, ratio: float) -> None:
        """
        Sub-samples the pointcloud.
        :param ratio: Fraction to keep.
        """
        selected_ind = np.random.choice(np.arange(0, self.nbr_points()), size=int(self.nbr_points() * ratio))
        self.points = self.points[:, selected_ind]

    def remove_close(self, radius: float) -> None:
        """
        Removes point too close within a certain radius from origin.
        :param radius: Radius below which points are removed.
        """

        x_filt = np.abs(self.points[0, :]) < radius
        y_filt = np.abs(self.points[1, :]) < radius
        not_close = np.logical_not(np.logical_and(x_filt, y_filt))
        self.points = self.points[:, not_close]

    def translate(self, x: np.ndarray) -> None:
        """
        Applies a translation to the point cloud.
        :param x: <np.float: 3, 1>. Translation in x, y, z.
        """
        for i in range(3):
            self.points[i, :] = self.points[i, :] + x[i]

    def rotate(self, rot_matrix: np.ndarray) -> None:
        """
        Applies a rotation.
        :param rot_matrix: <np.float: 3, 3>. Rotation matrix.
        """
        self.points[:3, :] = np.dot(rot_matrix, self.points[:3, :])

    def transform(self, transf_matrix: np.ndarray) -> None:
        """
        Applies a homogeneous transform.
        :param transf_matrix: <np.float: 4, 4>. Homogenous transformation matrix.
        """
        self.points[:3, :] = transf_matrix.dot(np.vstack((self.points[:3, :], np.ones(self.nbr_points()))))[:3, :]

    def render_height(self,
                      ax: Axes,
                      view: np.ndarray = np.eye(4),
                      x_lim: Tuple[float, float] = (-20, 20),
                      y_lim: Tuple[float, float] = (-20, 20),
                      marker_size: float = 1) -> None:
        """
        Very simple method that applies a transformation and then scatter plots the points colored by height (z-value).
        :param ax: Axes on which to render the points.
        :param view: <np.float: n, n>. Defines an arbitrary projection (n <= 4).
        :param x_lim: (min, max). x range for plotting.
        :param y_lim: (min, max). y range for plotting.
        :param marker_size: Marker size.
        """
        self._render_helper(2, ax, view, x_lim, y_lim, marker_size)

    def render_intensity(self,
                         ax: Axes,
                         view: np.ndarray = np.eye(4),
                         x_lim: Tuple[float, float] = (-20, 20),
                         y_lim: Tuple[float, float] = (-20, 20),
                         marker_size: float = 1) -> None:
        """
        Very simple method that applies a transformation and then scatter plots the points colored by intensity.
        :param ax: Axes on which to render the points.
        :param view: <np.float: n, n>. Defines an arbitrary projection (n <= 4).
        :param x_lim: (min, max).
        :param y_lim: (min, max).
        :param marker_size: Marker size.
        """
        self._render_helper(3, ax, view, x_lim, y_lim, marker_size)

    def _render_helper(self,
                       color_channel: int,
                       ax: Axes,
                       view: np.ndarray,
                       x_lim: Tuple[float, float],
                       y_lim: Tuple[float, float],
                       marker_size: float) -> None:
        """
        Helper function for rendering.
        :param color_channel: Point channel to use as color.
        :param ax: Axes on which to render the points.
        :param view: <np.float: n, n>. Defines an arbitrary projection (n <= 4).
        :param x_lim: (min, max).
        :param y_lim: (min, max).
        :param marker_size: Marker size.
        """
        points = view_points(self.points[:3, :], view, normalize=False)
        ax.scatter(points[0, :], points[1, :], c=self.points[color_channel, :], s=marker_size)
        ax.set_xlim(x_lim[0], x_lim[1])
        ax.set_ylim(y_lim[0], y_lim[1])

class LidarPointCloud(PointCloud):

    @staticmethod
    def nbr_dims() -> int:
        """
        Returns the number of dimensions.
        :return: Number of dimensions.
        """
        return 4

    @classmethod
    def from_file(cls, file_name: str) -> 'LidarPointCloud':
        """
        Loads LIDAR data from binary numpy format. Data is stored as (x, y, z, intensity, ring index).
        :param file_name: Path of the pointcloud file on disk.
        :return: LidarPointCloud instance (x, y, z, intensity).
        """

        assert file_name.endswith('.bin'), 'Unsupported filetype {}'.format(file_name)

        scan = np.fromfile(file_name, dtype=np.float32)
        points = scan.reshape((-1, 5))[:, :cls.nbr_dims()]
        return cls(points.T)


class RadarPointCloud(PointCloud):

    # Class-level settings for radar pointclouds, see from_file().
    invalid_states = [0]  # type: List[int]
    dynprop_states = range(7)  # type: List[int] # Use [0, 2, 6] for moving objects only.
    ambig_states = [3]  # type: List[int]

    @classmethod
    def disable_filters(cls) -> None:
        """
        Disable all radar filter settings.
        Use this method to plot all radar returns.
        Note that this method affects the global settings.
        """
        cls.invalid_states = list(range(18))
        cls.dynprop_states = list(range(8))
        cls.ambig_states = list(range(5))

    @classmethod
    def default_filters(cls) -> None:
        """
        Set the defaults for all radar filter settings.
        Note that this method affects the global settings.
        """
        cls.invalid_states = [0]
        cls.dynprop_states = range(7)
        cls.ambig_states = [3]

    @staticmethod
    def nbr_dims() -> int:
        """
        Returns the number of dimensions.
        :return: Number of dimensions.
        """
        return 18

    @classmethod
    def from_file(cls,
                  file_name: str,
                  invalid_states: List[int] = None,
                  dynprop_states: List[int] = None,
                  ambig_states: List[int] = None) -> 'RadarPointCloud':
        """
        Loads RADAR data from a Point Cloud Data file. See details below.
        :param file_name: The path of the pointcloud file.
        :param invalid_states: Radar states to be kept. See details below.
        :param dynprop_states: Radar states to be kept. Use [0, 2, 6] for moving objects only. See details below.
        :param ambig_states: Radar states to be kept. See details below.
        To keep all radar returns, set each state filter to range(18).
        :return: <np.float: d, n>. Point cloud matrix with d dimensions and n points.

        Example of the header fields:
        # .PCD v0.7 - Point Cloud Data file format
        VERSION 0.7
        FIELDS x y z dyn_prop id rcs vx vy vx_comp vy_comp is_quality_valid ambig_state x_rms y_rms invalid_state pdh0 vx_rms vy_rms
        SIZE 4 4 4 1 2 4 4 4 4 4 1 1 1 1 1 1 1 1
        TYPE F F F I I F F F F F I I I I I I I I
        COUNT 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
        WIDTH 125
        HEIGHT 1
        VIEWPOINT 0 0 0 1 0 0 0
        POINTS 125
        DATA binary

        Below some of the fields are explained in more detail:

        x is front, y is left

        vx, vy are the velocities in m/s.
        vx_comp, vy_comp are the velocities in m/s compensated by the ego motion.
        We recommend using the compensated velocities.

        invalid_state: state of Cluster validity state.
        (Invalid states)
        0x01	invalid due to low RCS
        0x02	invalid due to near-field artefact
        0x03	invalid far range cluster because not confirmed in near range
        0x05	reserved
        0x06	invalid cluster due to high mirror probability
        0x07	Invalid cluster because outside sensor field of view
        0x0d	reserved
        0x0e	invalid cluster because it is a harmonics
        (Valid states)
        0x00	valid
        0x04	valid cluster with low RCS
        0x08	valid cluster with azimuth correction due to elevation
        0x09	valid cluster with high child probability
        0x0a	valid cluster with high probability of being a 50 deg artefact
        0x0b	valid cluster but no local maximum
        0x0c	valid cluster with high artefact probability
        0x0f	valid cluster with above 95m in near range
        0x10	valid cluster with high multi-target probability
        0x11	valid cluster with suspicious angle

        dynProp: Dynamic property of cluster to indicate if is moving or not.
        0: moving
        1: stationary
        2: oncoming
        3: stationary candidate
        4: unknown
        5: crossing stationary
        6: crossing moving
        7: stopped

        ambig_state: State of Doppler (radial velocity) ambiguity solution.
        0: invalid
        1: ambiguous
        2: staggered ramp
        3: unambiguous
        4: stationary candidates

        pdh0: False alarm probability of cluster (i.e. probability of being an artefact caused by multipath or similar).
        0: invalid
        1: <25%
        2: 50%
        3: 75%
        4: 90%
        5: 99%
        6: 99.9%
        7: <=100%
        """

        assert file_name.endswith('.pcd'), 'Unsupported filetype {}'.format(file_name)

        meta = []
        with open(file_name, 'rb') as f:
            for line in f:
                line = line.strip().decode('utf-8')
                meta.append(line)
                if line.startswith('DATA'):
                    break

            data_binary = f.read()

        # Get the header rows and check if they appear as expected.
        assert meta[0].startswith('#'), 'First line must be comment'
        assert meta[1].startswith('VERSION'), 'Second line must be VERSION'
        sizes = meta[3].split(' ')[1:]
        types = meta[4].split(' ')[1:]
        counts = meta[5].split(' ')[1:]
        width = int(meta[6].split(' ')[1])
        height = int(meta[7].split(' ')[1])
        data = meta[10].split(' ')[1]
        feature_count = len(types)
        assert width > 0
        assert len([c for c in counts if c != c]) == 0, 'Error: COUNT not supported!'
        assert height == 1, 'Error: height != 0 not supported!'
        assert data == 'binary'

        # Lookup table for how to decode the binaries.
        unpacking_lut = {'F': {2: 'e', 4: 'f', 8: 'd'},
                         'I': {1: 'b', 2: 'h', 4: 'i', 8: 'q'},
                         'U': {1: 'B', 2: 'H', 4: 'I', 8: 'Q'}}
        types_str = ''.join([unpacking_lut[t][int(s)] for t, s in zip(types, sizes)])

        # Decode each point.
        offset = 0
        point_count = width
        points = []
        for i in range(point_count):
            point = []
            for p in range(feature_count):
                start_p = offset
                end_p = start_p + int(sizes[p])
                assert end_p < len(data_binary)
                point_p = struct.unpack(types_str[p], data_binary[start_p:end_p])[0]
                point.append(point_p)
                offset = end_p
            points.append(point)

        # A NaN in the first point indicates an empty pointcloud.
        point = np.array(points[0])
        if np.any(np.isnan(point)):
            return cls(np.zeros((feature_count, 0)))

        # Convert to numpy matrix.
        points = np.array(points).transpose()

        # If no parameters are provided, use default settings.
        invalid_states = cls.invalid_states if invalid_states is None else invalid_states
        dynprop_states = cls.dynprop_states if dynprop_states is None else dynprop_states
        ambig_states = cls.ambig_states if ambig_states is None else ambig_states

        # Filter points with an invalid state.
        valid = [p in invalid_states for p in points[-4, :]]
        points = points[:, valid]

        # Filter by dynProp.
        valid = [p in dynprop_states for p in points[3, :]]
        points = points[:, valid]

        # Filter by ambig_state.
        valid = [p in ambig_states for p in points[11, :]]
        points = points[:, valid]

        return cls(points)


class LidarSegPointCloud:
    """
    Class for a point cloud.
    """
    def __init__(self, points_path: str = None, labels_path: str = None):
        """
        Initialize a LidarSegPointCloud object.
        :param points_path: Path to the bin file containing the x, y, z and intensity of the points in the point cloud.
        :param labels_path: Path to the bin file containing the labels of the points in the point cloud.
        """
        self.points, self.labels = None, None
        if points_path:
            self.load_points(points_path)
        if labels_path:
            self.load_labels(labels_path)

    def load_points(self, path: str) -> None:
        """
        Loads the x, y, z and intensity of the points in the point cloud.
        :param path: Path to the bin file containing the x, y, z and intensity of the points in the point cloud.
        """
        self.points = LidarPointCloud.from_file(path).points.T  # [N, 4], where N is the number of points.
        if self.labels is not None:
            assert len(self.points) == len(self.labels), 'Error: There are {} points in the point cloud, ' \
                                                         'but {} labels'.format(len(self.points), len(self.labels))

    def load_labels(self, path: str) -> None:
        """
        Loads the labels of the points in the point cloud.
        :param path: Path to the bin file containing the labels of the points in the point cloud.
        """
        self.labels = load_bin_file(path)
        if self.points is not None:
            assert len(self.points) == len(self.labels), 'Error: There are {} points in the point cloud, ' \
                                                         'but {} labels'.format(len(self.points), len(self.labels))

    def render(self, name2color: Dict[str, Tuple[int]],
               name2id: Dict[str, int],
               ax: Axes,
               title: str = None,
               dot_size: int = 5) -> Axes:
        """
        Renders a point cloud onto an axes.
        :param name2color: The mapping from class name to class color.
        :param name2id: A dictionary containing the mapping from class names to class indices.
        :param ax: Axes onto which to render.
        :param title: Title of the plot.
        :param dot_size: Scatter plot dot size.
        :return: The axes onto which the point cloud has been rendered.
        """
        colors = colormap_to_colors(name2color, name2id)
        ax.scatter(self.points[:, 0], self.points[:, 1], c=colors[self.labels], s=dot_size)

        id2name = {idx: name for name, idx in name2id.items()}
        create_lidarseg_legend(self.labels, id2name, name2color, ax=ax)

        if title:
            ax.set_title(title)

        return ax

class TrackingRenderer:
    """
    Class that renders the tracking results in BEV and saves them to a folder.
    """
    def __init__(self,scene_id, save_path):
        """
        :param save_path:  Output path to save the renderings.
        """
        self.save_path = save_path
        self.id2color = {}  # The color of each track.
        self.scene_id=scene_id

    def render(self, nusc, events: DataFrame, timestamp: int, frame_gt: List[TrackingBox], frame_pred: List[TrackingBox]) \
            -> None:
        """
        Render function for a given scene timestamp
        :param events: motmetrics events for that particular
        :param timestamp: timestamp for the rendering
        :param frame_gt: list of ground truth boxes
        :param frame_pred: list of prediction boxes
        """
        # Init.
        #print('Rendering {}'.format(timestamp))
        switches = events[events.Type == 'SWITCH']
        switch_ids = switches.HId.values
        switch_dis = switches.D.values
        fp=events[events.Type == 'FP']
        fp_ids = fp.HId.values 
        fp_dis=fp.D.values
        fn=events[events.Type=='MISS']
        fn_ids = fn.OId.values
        match=events[events.Type=='MATCH']
        match_ids=match.HId.values
        match_dis=match.D.values

        fig, ax = plt.subplots(figsize=(12,12))
       
        #ground_truth = mpatches.Patch(color=(0/255, 0/255, 0/255), label='ground truth')
        #false_nagative = mpatches.Patch(color=(70/255, 130/255, 180/255), label='false negative')
        #ID_switch = mpatches.Patch(label='ID_switch, random colour')
        #matched = mpatches.Patch(label='matched, random colour with distance shown')
        
        #fig.legend(handles=[ground_truth, false_nagative, ID_switch, matched],loc='upper left',prop={'size': 20},ncol=1)

        # Plot GT boxes.
        for b in frame_gt:
            box = Box(b.ego_translation, b.size, Quaternion(b.rotation), name=b.tracking_name, token=b.tracking_id)
            #sample_token=b.sample_token
            
            if b.tracking_id in fn_ids:
                color = 'b'
                text='fn'
                center_bottom=box.center
                box.render(ax, view=np.eye(4), colors=(color, color, color), linewidth=1,linestyle='dashed')
                ax.text(center_bottom[0], center_bottom[1],text , fontsize = 15, color=color)
                #ax.scatter(center_bottom[0], center_bottom[1],color=color)
            else:
                color = 'k'
                box.render(ax, view=np.eye(4), colors=(color, color, color))

        # Plot predicted boxes.
        for b in frame_pred:
            box = Box(b.ego_translation, b.size, Quaternion(b.rotation), name=b.tracking_name, token=b.tracking_id)

            # Determine color for this tracking id.
            if b.tracking_id not in self.id2color.keys():
                self.id2color[b.tracking_id] = (float(hash(str(b.tracking_id)+ 'r') % 256) / 255,
                                                float(hash(str(b.tracking_id) + 'g') % 256) / 255,
                                                float(hash(str(b.tracking_id) + 'b') % 256) / 255)

            # Render box. Highlight identity switches in red.
            if b.tracking_id in switch_ids:
                color = self.id2color[b.tracking_id]
                idx=switch_ids.tolist().index(b.tracking_id)
                id=b.tracking_id
                text='id switch'+str(round(switch_dis[idx], 2))
                text2=str(round(b.tracking_score, 2))
                text3=str(id)
                box.render(ax, view=np.eye(4), colors=(color, color, color),linewidth=1)
                center_bottom=box.center
                ax.text(center_bottom[0], center_bottom[1]+1,text , fontsize = 15, color=color)
                ax.text(center_bottom[0], center_bottom[1],text3 , fontsize = 7, color=color)
                ax.text(center_bottom[0], center_bottom[1]-1,text2 , fontsize = 7, color=color)
                #ax.scatter(center_bottom[0], center_bottom[1],color=color)

            elif b.tracking_id in fp_ids:
                color = self.id2color[b.tracking_id]
                idx=fp_ids.tolist().index(b.tracking_id)
                id=b.tracking_id
                text='fp'
                text3=str(id)
                text2=str(round(b.tracking_score, 2))
                box.render(ax, view=np.eye(4), colors=(color, color, color),linewidth=1,linestyle=':')
                center_bottom=box.center
                ax.text(center_bottom[0], center_bottom[1]+1,text , fontsize = 15, color=color)
                ax.text(center_bottom[0], center_bottom[1],text3 , fontsize = 7, color=color)
                ax.text(center_bottom[0], center_bottom[1]-1,text2 , fontsize = 7, color=color)
                #ax.scatter(center_bottom[0], center_bottom[1],color=color)
            elif b.tracking_id in match_ids:
                color = self.id2color[b.tracking_id]
                idx=match_ids.tolist().index(b.tracking_id)
                id=match_ids[idx]
                text=str(round(match_dis[idx], 2))
                text3=str(id)
                text2=str(round(b.tracking_score, 2))
                box.render(ax, view=np.eye(4), colors=(color, color, color),linewidth=1)
                center_bottom=box.center
                ax.text(center_bottom[0], center_bottom[1]+1,text , fontsize = 15, color=color)
                ax.text(center_bottom[0], center_bottom[1],text3 , fontsize = 7, color=color)
                ax.text(center_bottom[0], center_bottom[1]-1,text2 , fontsize = 7, color=color)

            else:
                color = self.id2color[b.tracking_id]
                box.render(ax, view=np.eye(4), colors=(color, color, color), linestyle='dashed')

        '''
        # Plot lidar point cloud
        scenes = nusc.scene
        samples = nusc.sample
        sample_data=nusc.sample_data
        samples_for_this_scene=[x for x in samples if x['scene_token']==self.scene_id]
        for x in samples_for_this_scene:
            if x['timestamp']==timestamp:
                sample_token=x['token']
                break

        #sample_token = frame_gt[0].sample_token
        record = nusc.get('sample', sample_token)
        lidar_data = {}
        for channel, sample_data_channel_token in record['data'].items():
            if channel == 'LIDAR_TOP':
                sd_record = nusc.get('sample_data', sample_data_channel_token)
                lidar_data[channel] = sample_data_channel_token
                sample_rec = nusc.get('sample', sd_record['sample_token'])
                chan = sd_record['channel']
        pc = LidarPointCloud.from_file_multisweep(nusc, sample_rec, chan, 'LIDAR_TOP',nsweeps=1)

        #points = view_points(pc.points[:3, :], np.eye(4), normalize=False)
        points = pc.points[:2,:]
        colors = 'k'
        point_scale = 0.01
        ax.scatter(points[0, :], points[1, :], c=colors, s=point_scale)
        '''

        # Plot ego pose.
        plt.scatter(0, 0, s=96, facecolors='none', edgecolors='k', marker='o')
        plt.xlim(-50, 50)
        plt.ylim(-50, 50)

        # Save to disk and close figure.
        fig.savefig(os.path.join(self.save_path, '{}.png'.format(timestamp)))
        plt.close(fig)

def category_to_tracking_name(category_name: str) -> Optional[str]:
    """
    Default label mapping from nuScenes to nuScenes tracking classes.
    :param category_name: Generic nuScenes class.
    :return: nuScenes tracking class.
    """
    tracking_mapping = {
        'vehicle.bicycle': 'bicycle',
        'vehicle.bus.bendy': 'bus',
        'vehicle.bus.rigid': 'bus',
        'vehicle.car': 'car',
        'vehicle.motorcycle': 'motorcycle',
        'human.pedestrian.adult': 'pedestrian',
        'human.pedestrian.child': 'pedestrian',
        'human.pedestrian.construction_worker': 'pedestrian',
        'human.pedestrian.police_officer': 'pedestrian',
        'vehicle.trailer': 'trailer',
        'vehicle.truck': 'truck'
    }

    if category_name in tracking_mapping:
        return tracking_mapping[category_name]
    else:
        return None

def track_initialization_duration(df: DataFrame, obj_frequencies: DataFrame) -> float:
    """
    Computes the track initialization duration, which is the duration from the first occurrence of an object to
    it's first correct detection (TP).
    Note that this True Positive metric is undefined if there are no matched tracks.
    :param df: Motmetrics dataframe that is required, but not used here.
    :param obj_frequencies: Stores the GT tracking_ids and their frequencies.
    :return: The track initialization time.
    """
    tid = 0
    missed_tracks = 0
    for gt_tracking_id in obj_frequencies.index:
        # Get matches.
        dfo = df.noraw[df.noraw.OId == gt_tracking_id]
        notmiss = dfo[dfo.Type != 'MISS']

        if len(notmiss) == 0:
            # Consider only tracked objects.
            diff = 0
            missed_tracks += 1
        else:
            # Find the first time the object was detected and compute the difference to first time the object
            # entered the scene.
            diff = notmiss.index[0][0] - dfo.index[0][0]

        # Multiply number of sample differences with approx. sample period (0.5 sec).
        assert diff >= 0, 'Time difference should be larger than or equal to zero: %.2f'
        tid += diff * 0.5

    matched_tracks = len(obj_frequencies) - missed_tracks
    if matched_tracks == 0:
        # Return nan if there are no matches.
        return np.nan
    else:
        return tid / matched_tracks


def longest_gap_duration(df: DataFrame, obj_frequencies: DataFrame) -> float:
    """
    Computes the longest gap duration, which is the longest duration of any gaps in the detection of an object.
    Note that this True Positive metric is undefined if there are no matched tracks.
    :param df: Motmetrics dataframe that is required, but not used here.
    :param obj_frequencies: Dataframe with all object frequencies.
    :return: The longest gap duration.
    """
    # Return nan if the class is not in the GT.
    if len(obj_frequencies.index) == 0:
        return np.nan

    lgd = 0
    missed_tracks = 0
    for gt_tracking_id in obj_frequencies.index:
        # Find the frame_ids object is tracked and compute the gaps between those. Take the maximum one for longest gap.
        dfo = df.noraw[df.noraw.OId == gt_tracking_id]
        matched = set(dfo[dfo.Type != 'MISS'].index.get_level_values(0).values)

        if len(matched) == 0:
            # Ignore untracked objects.
            gap = 0
            missed_tracks += 1
        else:
            # Find the biggest gap.
            # Note that we don't need to deal with FPs within the track as the GT is interpolated.
            gap = 0  # The biggest gap found.
            cur_gap = 0  # Current gap.
            first_index = dfo.index[0][0]
            last_index = dfo.index[-1][0]

            for i in range(first_index, last_index + 1):
                if i in matched:
                    # Reset when matched.
                    gap = np.maximum(gap, cur_gap)
                    cur_gap = 0
                else:  # Grow gap when missed.
                    # Gap grows.
                    cur_gap += 1

            gap = np.maximum(gap, cur_gap)

        # Multiply number of sample differences with approx. sample period (0.5 sec).
        assert gap >= 0, 'Time difference should be larger than or equal to zero: %.2f'
        lgd += gap * 0.5

    # Average LGD over the number of tracks.
    matched_tracks = len(obj_frequencies) - missed_tracks
    if matched_tracks == 0:
        # Return nan if there are no matches.
        lgd = np.nan
    else:
        lgd = lgd / matched_tracks

    return lgd


def motar(df: DataFrame, num_matches: int, num_misses: int, num_switches: int, num_false_positives: int,
          num_objects: int, alpha: float = 1.0) -> float:
    """
    Initializes a MOTAR class which refers to the modified MOTA metric at https://www.nuscenes.org/tracking.
    Note that we use the measured recall, which is not identical to the hypothetical recall of the
    AMOTA/AMOTP thresholds.
    :param df: Motmetrics dataframe that is required, but not used here.
    :param num_matches: The number of matches, aka. false positives.
    :param num_misses: The number of misses, aka. false negatives.
    :param num_switches: The number of identity switches.
    :param num_false_positives: The number of false positives.
    :param num_objects: The total number of objects of this class in the GT.
    :param alpha: MOTAR weighting factor (previously 0.2).
    :return: The MOTAR or nan if there are no GT objects.
    """
    recall = num_matches / num_objects
    nominator = (num_misses + num_switches + num_false_positives) - (1 - recall) * num_objects
    denominator = recall * num_objects
    if denominator == 0:
        motar_val = np.nan
    else:
        motar_val = 1 - alpha * nominator / denominator
        motar_val = np.maximum(0, motar_val)

    return motar_val


def mota_custom(df: DataFrame, num_misses: int, num_switches: int, num_false_positives: int, num_objects: int) -> float:
    """
    Multiple object tracker accuracy.
    Based on py-motmetric's mota function.
    Compared to the original MOTA definition, we clip values below 0.
    :param df: Motmetrics dataframe that is required, but not used here.
    :param num_misses: The number of misses, aka. false negatives.
    :param num_switches: The number of identity switches.
    :param num_false_positives: The number of false positives.
    :param num_objects: The total number of objects of this class in the GT.
    :return: The MOTA or 0 if below 0.
    """
    mota = 1. - (num_misses + num_switches + num_false_positives) / num_objects
    mota = np.maximum(0, mota)
    return mota


def motp_custom(df: DataFrame, num_detections: float) -> float:
    """
    Multiple object tracker precision.
    Based on py-motmetric's motp function.
    Additionally we check whether there are any detections.
    :param df: Motmetrics dataframe that is required, but not used here.
    :param num_detections: The number of detections.
    :return: The MOTP or 0 if there are no detections.
    """
    # Note that the default motmetrics function throws a warning when num_detections == 0.
    if num_detections == 0:
        return np.nan
    return df.noraw['D'].sum() / num_detections


def faf(df: DataFrame, num_false_positives: float, num_frames: float) -> float:
    """
    The average number of false alarms per frame.
    :param df: Motmetrics dataframe that is required, but not used here.
    :param num_false_positives: The number of false positives.
    :param num_frames: The number of frames.
    :return: Average FAF.
    """
    return num_false_positives / num_frames * 100


def num_fragmentations_custom(df: DataFrame, obj_frequencies: DataFrame) -> float:
    """
    Total number of switches from tracked to not tracked.
    Based on py-motmetric's num_fragmentations function.
    :param df: Motmetrics dataframe that is required, but not used here.
    :param obj_frequencies: Stores the GT tracking_ids and their frequencies.
    :return: The number of fragmentations.
    """
    fra = 0
    for o in obj_frequencies.index:
        # Find first and last time object was not missed (track span). Then count
        # the number switches from NOT MISS to MISS state.
        dfo = df.noraw[df.noraw.OId == o]
        notmiss = dfo[dfo.Type != 'MISS']
        if len(notmiss) == 0:
            continue
        first = notmiss.index[0]
        last = notmiss.index[-1]
        diffs = dfo.loc[first:last].Type.apply(lambda x: 1 if x == 'MISS' else 0).diff()
        fra += diffs[diffs == 1].count()

    return fra

def metric_name_to_print_format(metric_name) -> str:
    """
    Get the standard print format (numerical precision) for each metric.
    :param metric_name: The lowercase metric name.
    :return: The print format.
    """
    if metric_name in ['amota', 'amotp', 'motar', 'recall', 'mota', 'motp']:
        print_format = '%.3f'
    elif metric_name in ['tid', 'lgd']:
        print_format = '%.2f'
    elif metric_name in ['faf']:
        print_format = '%.1f'
    else:
        print_format = '%d'
    return print_format


def print_final_metrics(metrics: TrackingMetrics, result = None) -> None:
    """
    Print metrics to stdout.
    :param metrics: The output of evaluate().
    """
    print('\n### Final results ###')

    # Print per-class metrics.
    metric_names = metrics.label_metrics.keys()
    print('\nPer-class results:')
    print('\t\t', end='')
    print('\t'.join([m.upper() for m in metric_names]))

    class_names = metrics.class_names
    max_name_length = 7
    for class_name in class_names:
        print_class_name = class_name[:max_name_length].ljust(max_name_length + 1)
        print('%s' % print_class_name, end='')

        for metric_name in metric_names:
            val = metrics.label_metrics[metric_name][class_name]
            print_format = '%f' if np.isnan(val) else metric_name_to_print_format(metric_name)
            print('\t%s' % (print_format % val), end='')

        print()

        if result != None:
            result[class_name]={}
            for metric_name in metric_names:
                val = metrics.label_metrics[metric_name][class_name]
                result[class_name][metric_name]=val
            return result

    # Print high-level metrics.
    print('\nAggregated results:')
    for metric_name in metric_names:
        val = metrics.compute_metric(metric_name, 'all')
        print_format = metric_name_to_print_format(metric_name)
        print('%s\t%s' % (metric_name.upper(), print_format % val))

    print('Eval time: %.1fs' % metrics.eval_time)
    print()
        
def print_threshold_metrics(metrics: Dict[str, Dict[str, float]]) -> None:
    """
    Print only a subset of the metrics for the current class and threshold.
    :param metrics: A dictionary representation of the metrics.
    """
    # Specify threshold name and metrics.
    assert len(metrics['mota_custom'].keys()) == 1
    threshold_str = list(metrics['mota_custom'].keys())[0]
    mota=metrics['mota_custom'][threshold_str]
    motar_val = metrics['motar'][threshold_str]
    motp = metrics['motp_custom'][threshold_str]
    recall = metrics['recall'][threshold_str]
    num_frames = metrics['num_frames'][threshold_str]
    num_objects = metrics['num_objects'][threshold_str]
    num_predictions = metrics['num_predictions'][threshold_str]
    num_false_positives = metrics['num_false_positives'][threshold_str]
    num_misses = metrics['num_misses'][threshold_str]
    num_switches = metrics['num_switches'][threshold_str]
    num_matches = metrics['num_matches'][threshold_str]

    # Print.
    print('%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s'
          % ('\t','MOTAR','MOTA' ,'MOTP', 'Recall', 'Frames',
             'GT', 'GT-Mtch', 'GT-Miss', 'GT-IDS',
             'Pred', 'Pred-TP', 'Pred-FP', 'Pred-IDS',))
    print('%s\t%.3f\t%.3f\t%.3f\t%.3f\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d'
          % (threshold_str, motar_val,mota, motp, recall, num_frames,
             num_objects, num_matches, num_misses, num_switches,
             num_predictions, num_matches, num_false_positives, num_switches))
    #print('MOTA {}'.format(str(round(mota,2))))
    print()

    # Check metrics for consistency.
    assert num_objects == num_matches + num_misses + num_switches
    assert num_predictions == num_matches + num_false_positives + num_switches

def create_motmetrics() -> MetricsHost:
    """
    Creates a MetricsHost and populates it with default and custom metrics.
    It does not populate the global metrics which are more time consuming.
    :return The initialized MetricsHost object with default MOT metrics.
    """
    # Create new metrics host object.
    mh = MetricsHost()

    # Suppress deprecation warning from py-motmetrics.
    warnings.filterwarnings('ignore', category=DeprecationWarning)

    # Register standard metrics.
    fields = [
        'num_frames', 'obj_frequencies', 'num_matches', 'num_switches', 'num_false_positives', 'num_misses',
        'num_detections', 'num_objects', 'num_predictions', 'mostly_tracked', 'mostly_lost', 'num_fragmentations',
        'motp', 'mota', 'precision', 'recall', 'track_ratios'
    ]
    for field in fields:
        mh.register(getattr(motmetrics.metrics, field), formatter='{:d}'.format)

    # Reenable deprecation warning.
    warnings.filterwarnings('default', category=DeprecationWarning)

    # Register custom metrics.
    # Specify all inputs to avoid errors incompatibility between type hints and py-motmetric's introspection.
    mh.register(motar, ['num_matches', 'num_misses', 'num_switches', 'num_false_positives', 'num_objects'],
                formatter='{:.2%}'.format, name='motar')
    mh.register(mota_custom, ['num_misses', 'num_switches', 'num_false_positives', 'num_objects'],
                formatter='{:.2%}'.format, name='mota_custom')
    mh.register(motp_custom, ['num_detections'],
                formatter='{:.2%}'.format, name='motp_custom')
    mh.register(num_fragmentations_custom, ['obj_frequencies'],
                formatter='{:.2%}'.format, name='num_fragmentations_custom')
    mh.register(faf, ['num_false_positives', 'num_frames'],
                formatter='{:.2%}'.format, name='faf')
    mh.register(track_initialization_duration, ['obj_frequencies'],
                formatter='{:.2%}'.format, name='tid')
    mh.register(longest_gap_duration, ['obj_frequencies'],
                formatter='{:.2%}'.format, name='lgd')

    return mh

def render_for_the_best_threshold(nusc, class_name, threshold, tracks_gt, tracks_pred,threshold_render_classes,threshold_output_dir,dist_th_tp):
    # Groundtruth and tracker contain lists for every single frame containing lists detections.
    for scene_id in tracks_gt.keys():
        # Initialize accumulator and frame_id for this scene
        acc = MOTAccumulatorCustom()
        frame_id = 0  # Frame ids must be unique across all scenes
        # Retrieve GT and preds.
        scene_tracks_gt = tracks_gt[scene_id]
        scene_tracks_pred = tracks_pred[scene_id]
        # Visualize the boxes in this frame.
        if class_name in threshold_render_classes:
            save_path = os.path.join(threshold_output_dir,'threshold',str(scene_id), class_name, str(round(threshold,2)))
            os.makedirs(save_path, exist_ok=True)
            renderer = TrackingRenderer(scene_id, save_path)
        else:
            renderer = None
        for timestamp in scene_tracks_gt.keys():
            # Select only the current class.
            frame_gt = scene_tracks_gt[timestamp]
            frame_pred = scene_tracks_pred[timestamp]
            frame_gt = [f for f in frame_gt if f.tracking_name == class_name]
            frame_pred = [f for f in frame_pred if f.tracking_name == class_name and f.tracking_score >= threshold]
            # Abort if there are neither GT nor pred boxes.
            gt_ids = [gg.tracking_id for gg in frame_gt]
            pred_ids = [tt.tracking_id for tt in frame_pred]
            if len(gt_ids) == 0 and len(pred_ids) == 0:
                continue
            # Calculate distances.
            # Note that the distance function is hard-coded to achieve significant speedups via vectorization.
            if len(frame_gt) == 0 or len(frame_pred) == 0:
                distances = np.ones((0, 0))
            else:
                gt_boxes = np.array([b.translation[:2] for b in frame_gt])
                pred_boxes = np.array([b.translation[:2]
                                      for b in frame_pred])
                distances = sklearn.metrics.pairwise.euclidean_distances(
                    gt_boxes, pred_boxes)
            # Distances that are larger than the threshold won't be associated.
            assert len(distances) == 0 or not np.all(np.isnan(distances))
            distances[distances >= dist_th_tp] = np.nan
            # Accumulate results.
            # Note that we cannot use timestamp as frameid as motmetrics assumes it's an integer.
            acc.update(gt_ids, pred_ids, distances, frameid=frame_id)
            # Store scores of matches, which are used to determine recall thresholds.
            events = acc.events.loc[frame_id]
            # Render the boxes in this frame.
            if class_name in threshold_render_classes:
                renderer.render(nusc,events, timestamp, frame_gt, frame_pred)
            frame_id += 1

def load_prediction(result_path: str, max_boxes_per_sample: int, box_cls, verbose: bool = False) \
        -> Tuple[EvalBoxes, Dict]:
    """
    Loads object predictions from file.
    :param result_path: Path to the .json result file provided by the user.
    :param max_boxes_per_sample: Maximim number of boxes allowed per sample.
    :param box_cls: Type of box to load, e.g. DetectionBox or TrackingBox.
    :param verbose: Whether to print messages to stdout.
    :return: The deserialized results and meta data.
    """

    # Load from file and check that the format is correct.
    with open(result_path) as f:
        data = json.load(f)
    assert 'results' in data, 'Error: No field `results` in result file. Please note that the result format changed.' \
                              'See https://www.nuscenes.org/object-detection for more information.'

    # Deserialize results and get meta data.
    all_results = EvalBoxes.deserialize(data['results'], box_cls)
    meta = data['meta']
    if verbose:
        print("Loaded results from {}. Found detections for {} samples."
              .format(result_path, len(all_results.sample_tokens)))

    # Check that each sample has no more than x predicted boxes.
    for sample_token in all_results.sample_tokens:
        assert len(all_results.boxes[sample_token]) <= max_boxes_per_sample, \
            "Error: Only <= %d boxes per sample allowed!" % max_boxes_per_sample

    return all_results, meta

def category_to_detection_name(category_name: str) -> Optional[str]:
    """
    Default label mapping from nuScenes to nuScenes detection classes.
    Note that pedestrian does not include personal_mobility, stroller and wheelchair.
    :param category_name: Generic nuScenes class.
    :return: nuScenes detection class.
    """
    detection_mapping = {
        'movable_object.barrier': 'barrier',
        'vehicle.bicycle': 'bicycle',
        'vehicle.bus.bendy': 'bus',
        'vehicle.bus.rigid': 'bus',
        'vehicle.car': 'car',
        'vehicle.construction': 'construction_vehicle',
        'vehicle.motorcycle': 'motorcycle',
        'human.pedestrian.adult': 'pedestrian',
        'human.pedestrian.child': 'pedestrian',
        'human.pedestrian.construction_worker': 'pedestrian',
        'human.pedestrian.police_officer': 'pedestrian',
        'movable_object.trafficcone': 'traffic_cone',
        'vehicle.trailer': 'trailer',
        'vehicle.truck': 'truck'
    }

    if category_name in detection_mapping:
        return detection_mapping[category_name]
    else:
        return None

def detection_name_to_rel_attributes(detection_name: str) -> List[str]:
    """
    Returns a list of relevant attributes for a given detection class.
    :param detection_name: The detection class.
    :return: List of relevant attributes.
    """
    if detection_name in ['pedestrian']:
        rel_attributes = ['pedestrian.moving', 'pedestrian.sitting_lying_down', 'pedestrian.standing']
    elif detection_name in ['bicycle', 'motorcycle']:
        rel_attributes = ['cycle.with_rider', 'cycle.without_rider']
    elif detection_name in ['car', 'bus', 'construction_vehicle', 'trailer', 'truck']:
        rel_attributes = ['vehicle.moving', 'vehicle.parked', 'vehicle.stopped']
    elif detection_name in ['barrier', 'traffic_cone']:
        # Classes without attributes: barrier, traffic_cone.
        rel_attributes = []
    else:
        raise ValueError('Error: %s is not a valid detection class.' % detection_name)

    return rel_attributes

def load_gt(nusc: NuScenes, eval_split: str, box_cls, verbose: bool = False) -> EvalBoxes:
    """
    Loads ground truth boxes from DB.
    :param nusc: A NuScenes instance.
    :param eval_split: The evaluation split for which we load GT boxes.
    :param box_cls: Type of box to load, e.g. DetectionBox or TrackingBox.
    :param verbose: Whether to print messages to stdout.
    :return: The GT boxes.
    """
    # Init.
    if box_cls == DetectionBox:
        attribute_map = {a['token']: a['name'] for a in nusc.attribute}

    if verbose:
        print('Loading annotations for {} split from nuScenes version: {}'.format(eval_split, nusc.version))
    # Read out all sample_tokens in DB.
    sample_tokens_all = [s['token'] for s in nusc.sample]
    assert len(sample_tokens_all) > 0, "Error: Database has no samples!"

    # Only keep samples from this split.
    splits = create_splits_scenes()

    # Check compatibility of split with nusc_version.
    version = nusc.version
    if eval_split in {'train', 'val', 'train_detect', 'train_track'}:
        assert version.endswith('trainval'), \
            'Error: Requested split {} which is not compatible with NuScenes version {}'.format(eval_split, version)
    elif eval_split in {'mini_train', 'mini_val'}:
        assert version.endswith('mini'), \
            'Error: Requested split {} which is not compatible with NuScenes version {}'.format(eval_split, version)
    elif eval_split == 'test':
        assert version.endswith('test'), \
            'Error: Requested split {} which is not compatible with NuScenes version {}'.format(eval_split, version)
    else:
        raise ValueError('Error: Requested split {} which this function cannot map to the correct NuScenes version.'
                         .format(eval_split))

    if eval_split == 'test':
        # Check that you aren't trying to cheat :).
        assert len(nusc.sample_annotation) > 0, \
            'Error: You are trying to evaluate on the test set but you do not have the annotations!'

    sample_tokens = []
    for sample_token in sample_tokens_all:
        scene_token = nusc.get('sample', sample_token)['scene_token']
        scene_record = nusc.get('scene', scene_token)
        if scene_record['name'] in splits[eval_split]:
            sample_tokens.append(sample_token)

    all_annotations = EvalBoxes()

    # Load annotations and filter predictions and annotations.
    tracking_id_set = set()
    for sample_token in sample_tokens:

        sample = nusc.get('sample', sample_token)
        sample_annotation_tokens = sample['anns']

        sample_boxes = []
        for sample_annotation_token in sample_annotation_tokens:

            sample_annotation = nusc.get('sample_annotation', sample_annotation_token)
            if box_cls == DetectionBox:
                # Get label name in detection task and filter unused labels.
                detection_name = category_to_detection_name(sample_annotation['category_name'])
                if detection_name is None:
                    continue

                # Get attribute_name.
                attr_tokens = sample_annotation['attribute_tokens']
                attr_count = len(attr_tokens)
                if attr_count == 0:
                    attribute_name = ''
                elif attr_count == 1:
                    attribute_name = attribute_map[attr_tokens[0]]
                else:
                    raise Exception('Error: GT annotations must not have more than one attribute!')

                sample_boxes.append(
                    box_cls(
                        sample_token=sample_token,
                        translation=sample_annotation['translation'],
                        size=sample_annotation['size'],
                        rotation=sample_annotation['rotation'],
                        velocity=nusc.box_velocity(sample_annotation['token'])[:2],
                        num_pts=sample_annotation['num_lidar_pts'] + sample_annotation['num_radar_pts'],
                        detection_name=detection_name,
                        detection_score=-1.0,  # GT samples do not have a score.
                        attribute_name=attribute_name
                    )
                )
            elif box_cls == TrackingBox:
                # Use nuScenes token as tracking id.
                tracking_id = sample_annotation['instance_token']
                tracking_id_set.add(tracking_id)

                # Get label name in detection task and filter unused labels.
                # Import locally to avoid errors when motmetrics package is not installed.
                tracking_name = category_to_tracking_name(sample_annotation['category_name'])
                if tracking_name is None:
                    continue

                sample_boxes.append(
                    box_cls(
                        sample_token=sample_token,
                        translation=sample_annotation['translation'],
                        size=sample_annotation['size'],
                        rotation=sample_annotation['rotation'],
                        velocity=nusc.box_velocity(sample_annotation['token'])[:2],
                        num_pts=sample_annotation['num_lidar_pts'] + sample_annotation['num_radar_pts'],
                        tracking_id=tracking_id,
                        tracking_name=tracking_name,
                        tracking_score=-1.0  # GT samples do not have a score.
                    )
                )
            else:
                raise NotImplementedError('Error: Invalid box_cls %s!' % box_cls)

        all_annotations.add_boxes(sample_token, sample_boxes)

    if verbose:
        print("Loaded ground truth annotations for {} samples.".format(len(all_annotations.sample_tokens)))

    return all_annotations

def add_center_dist(nusc: NuScenes,
                    eval_boxes: EvalBoxes):
    """
    Adds the cylindrical (xy) center distance from ego vehicle to each box.
    :param nusc: The NuScenes instance.
    :param eval_boxes: A set of boxes, either GT or predictions.
    :return: eval_boxes augmented with center distances.
    """
    for sample_token in eval_boxes.sample_tokens:
        sample_rec = nusc.get('sample', sample_token)
        sd_record = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
        pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

    
        for box in eval_boxes[sample_token]:
            
            # Both boxes and ego pose are given in global coord system, so distance can be calculated directly.
            # Note that the z component of the ego pose is 0.
            ego_translation = (box.translation[0] - pose_record['translation'][0],
                               box.translation[1] - pose_record['translation'][1],
                               box.translation[2] - pose_record['translation'][2])
            if isinstance(box, DetectionBox) or isinstance(box, TrackingBox):
                box.ego_translation = ego_translation
            else:
                raise NotImplementedError

    return eval_boxes


def filter_eval_boxes(nusc: NuScenes,
                      eval_boxes: EvalBoxes,
                      max_dist: Dict[str, float],
                      verbose: bool = False) -> EvalBoxes:
    """
    Applies filtering to boxes. Distance, bike-racks and points per box.
    :param nusc: An instance of the NuScenes class.
    :param eval_boxes: An instance of the EvalBoxes class.
    :param max_dist: Maps the detection name to the eval distance threshold for that class.
    :param verbose: Whether to print to stdout.
    """
    # Retrieve box type for detectipn/tracking boxes.
    class_field = _get_box_class_field(eval_boxes)

    # Accumulators for number of filtered boxes.
    total, dist_filter, point_filter, bike_rack_filter = 0, 0, 0, 0
    for ind, sample_token in enumerate(eval_boxes.sample_tokens):

        # Filter on distance first.
        total += len(eval_boxes[sample_token])
        eval_boxes.boxes[sample_token] = [box for box in eval_boxes[sample_token] if
                                          box.ego_dist < max_dist[box.__getattribute__(class_field)]]
        dist_filter += len(eval_boxes[sample_token])

        # Then remove boxes with zero points in them. Eval boxes have -1 points by default.
        eval_boxes.boxes[sample_token] = [box for box in eval_boxes[sample_token] if not box.num_pts == 0]
        point_filter += len(eval_boxes[sample_token])

        # Perform bike-rack filtering.
        sample_anns = nusc.get('sample', sample_token)['anns']
        bikerack_recs = [nusc.get('sample_annotation', ann) for ann in sample_anns if
                         nusc.get('sample_annotation', ann)['category_name'] == 'static_object.bicycle_rack']
        bikerack_boxes = [Box(rec['translation'], rec['size'], Quaternion(rec['rotation'])) for rec in bikerack_recs]
        filtered_boxes = []
        for box in eval_boxes[sample_token]:
            if box.__getattribute__(class_field) in ['bicycle', 'motorcycle']:
                in_a_bikerack = False
                for bikerack_box in bikerack_boxes:
                    if np.sum(points_in_box(bikerack_box, np.expand_dims(np.array(box.translation), axis=1))) > 0:
                        in_a_bikerack = True
                if not in_a_bikerack:
                    filtered_boxes.append(box)
            else:
                filtered_boxes.append(box)

        eval_boxes.boxes[sample_token] = filtered_boxes
        bike_rack_filter += len(eval_boxes.boxes[sample_token])

    if verbose:
        print("=> Original number of boxes: %d" % total)
        print("=> After distance based filtering: %d" % dist_filter)
        print("=> After LIDAR and RADAR points based filtering: %d" % point_filter)
        print("=> After bike rack filtering: %d" % bike_rack_filter)

    return eval_boxes


def _get_box_class_field(eval_boxes: EvalBoxes) -> str:
    """
    Retrieve the name of the class field in the boxes.
    This parses through all boxes until it finds a valid box.
    If there are no valid boxes, this function throws an exception.
    :param eval_boxes: The EvalBoxes used for evaluation.
    :return: The name of the class field in the boxes, e.g. detection_name or tracking_name.
    """
    assert len(eval_boxes.boxes) > 0
    box = None
    for val in eval_boxes.boxes.values():
        if len(val) > 0:
            box = val[0]
            break
    if isinstance(box, DetectionBox):
        class_field = 'detection_name'
    elif isinstance(box, TrackingBox):
        class_field = 'tracking_name'
    else:
        raise Exception('Error: Invalid box type: %s' % box)

    return class_field
def interpolate_tracking_boxes(left_box: TrackingBox, right_box: TrackingBox, right_ratio: float) -> TrackingBox:
    """
    Linearly interpolate box parameters between two boxes.
    :param left_box: A Trackingbox.
    :param right_box: Another TrackingBox
    :param right_ratio: Weight given to the right box.
    :return: The interpolated TrackingBox.
    """
    def interp_list(left, right, rratio):
        return tuple(
            (1.0 - rratio) * np.array(left, dtype=float)
            + rratio * np.array(right, dtype=float)
        )

    def interp_float(left, right, rratio):
        return (1.0 - rratio) * float(left) + rratio * float(right)

    # Interpolate quaternion.
    rotation = Quaternion.slerp(
        q0=Quaternion(left_box.rotation),
        q1=Quaternion(right_box.rotation),
        amount=right_ratio
    ).elements

    # Score will remain -1 for GT.
    tracking_score = interp_float(left_box.tracking_score, right_box.tracking_score, right_ratio)

    return TrackingBox(sample_token=right_box.sample_token,
                       translation=interp_list(left_box.translation, right_box.translation, right_ratio),
                       size=interp_list(left_box.size, right_box.size, right_ratio),
                       rotation=rotation,
                       velocity=interp_list(left_box.velocity, right_box.velocity, right_ratio),
                       ego_translation=interp_list(left_box.ego_translation, right_box.ego_translation,
                                                   right_ratio),  # May be inaccurate.
                       tracking_id=right_box.tracking_id,
                       tracking_name=right_box.tracking_name,
                       tracking_score=tracking_score)


def interpolate_tracks(tracks_by_timestamp: DefaultDict[int, List[TrackingBox]]) -> DefaultDict[int, List[TrackingBox]]:
    """
    Interpolate the tracks to fill in holes, especially since GT boxes with 0 lidar points are removed.
    This interpolation does not take into account visibility. It interpolates despite occlusion.
    :param tracks_by_timestamp: The tracks.
    :return: The interpolated tracks.
    """
    # Group tracks by id.
    tracks_by_id = defaultdict(list)
    track_timestamps_by_id = defaultdict(list)
    for timestamp, tracking_boxes in tracks_by_timestamp.items():
        for tracking_box in tracking_boxes:
            tracks_by_id[tracking_box.tracking_id].append(tracking_box)
            track_timestamps_by_id[tracking_box.tracking_id].append(timestamp)

    # Interpolate missing timestamps for each track.
    timestamps = tracks_by_timestamp.keys()
    interpolate_count = 0
    for timestamp in timestamps:
        for tracking_id, track in tracks_by_id.items():
            if track_timestamps_by_id[tracking_id][0] <= timestamp <= track_timestamps_by_id[tracking_id][-1] and \
                    timestamp not in track_timestamps_by_id[tracking_id]:

                # Find the closest boxes before and after this timestamp.
                right_ind = bisect(track_timestamps_by_id[tracking_id], timestamp)
                left_ind = right_ind - 1
                right_timestamp = track_timestamps_by_id[tracking_id][right_ind]
                left_timestamp = track_timestamps_by_id[tracking_id][left_ind]
                right_tracking_box = tracks_by_id[tracking_id][right_ind]
                left_tracking_box = tracks_by_id[tracking_id][left_ind]
                right_ratio = float(right_timestamp - timestamp) / (right_timestamp - left_timestamp)

                # Interpolate.
                tracking_box = interpolate_tracking_boxes(left_tracking_box, right_tracking_box, right_ratio)
                interpolate_count += 1
                tracks_by_timestamp[timestamp].append(tracking_box)

    return tracks_by_timestamp

def create_tracks(all_boxes: EvalBoxes, nusc: NuScenes, eval_split: str, gt: bool) \
        -> Dict[str, Dict[int, List[TrackingBox]]]:
    """
    Returns all tracks for all scenes. Samples within a track are sorted in chronological order.
    This can be applied either to GT or predictions.
    :param all_boxes: Holds all GT or predicted boxes.
    :param nusc: The NuScenes instance to load the sample information from.
    :param eval_split: The evaluation split for which we create tracks.
    :param gt: Whether we are creating tracks for GT or predictions
    :return: The tracks.
    """
    # Only keep samples from this split.
    splits = create_splits_scenes()
    scene_tokens = set()
    for sample_token in all_boxes.sample_tokens:
        scene_token = nusc.get('sample', sample_token)['scene_token']
        scene = nusc.get('scene', scene_token)
        if scene['name'] in splits[eval_split]:
            scene_tokens.add(scene_token)

    # Tracks are stored as dict {scene_token: {timestamp: List[TrackingBox]}}.
    tracks = defaultdict(lambda: defaultdict(list))

    # Init all scenes and timestamps to guarantee completeness.
    for scene_token in scene_tokens:
        # Init all timestamps in this scene.
        scene = nusc.get('scene', scene_token)
        cur_sample_token = scene['first_sample_token']
        while True:
            # Initialize array for current timestamp.
            cur_sample = nusc.get('sample', cur_sample_token)
            tracks[scene_token][cur_sample['timestamp']] = []
            #tracks[scene_token][cur_sample_token]=[]

            # Abort after the last sample.
            if cur_sample_token == scene['last_sample_token']:
                break

            # Move to next sample.
            cur_sample_token = cur_sample['next']

    # Group annotations wrt scene and timestamp.
    for sample_token in all_boxes.sample_tokens:
        sample_record = nusc.get('sample', sample_token)
        scene_token = sample_record['scene_token']
        tracks[scene_token][sample_record['timestamp']] = all_boxes.boxes[sample_token]
        #tracks[scene_token][sample_token] = all_boxes.boxes[sample_token]


    # Replace box scores with track score (average box score). This only affects the compute_thresholds method and
    # should be done before interpolation to avoid diluting the original scores with interpolated boxes.
    if not gt:
        for scene_id, scene_tracks in tracks.items():
            # For each track_id, collect the scores.
            track_id_scores = defaultdict(list)
            for timestamp, boxes in scene_tracks.items():
                for box in boxes:
                    track_id_scores[box.tracking_id].append(box.tracking_score)

            # Compute average scores for each track.
            track_id_avg_scores = {}
            for tracking_id, scores in track_id_scores.items():
                track_id_avg_scores[tracking_id] = np.mean(scores)

            # Apply average score to each box.
            for timestamp, boxes in scene_tracks.items():
                for box in boxes:
                    box.tracking_score = track_id_avg_scores[box.tracking_id]

    # Interpolate GT and predicted tracks.
    for scene_token in tracks.keys():
        tracks[scene_token] = interpolate_tracks(tracks[scene_token])

        if not gt:
            # Make sure predictions are sorted in in time. (Always true for GT).
            tracks[scene_token] = defaultdict(list, sorted(tracks[scene_token].items(), key=lambda kv: kv[0]))

    return tracks
