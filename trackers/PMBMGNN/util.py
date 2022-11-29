import numpy as np
import copy
from scipy.stats import multivariate_normal
from scipy.stats import poisson
import math
import numpy.matlib
import matplotlib.pyplot as plt
import time
from functools import reduce
import operator
import pickle
import argparse
from scipy.optimize import linear_sum_assignment

def gen_filter_model(average_number_of_clutter_per_frame, p_S, p_D,classification, extraction_thr, ber_thr, poi_thr, eB_thr,ber_gating,use_ds_as_pd, P_init,use_giou=False, gating_mode='mahalanobis'):

    Q = {
        'bicycle': {'x': 1.98881347e-02, 'y': 1.36552276e-02, 'z': 5.10175742e-03, 'yaw': 1.33430252e-01,
                    'l': 0, 'w': 0, 'h': 0,
                    'dx': 1.98881347e-02, 'dy': 1.36552276e-02, 'dz': 5.10175742e-03, 'dyaw': 1.33430252e-01},
        'bus': {'x': 1.17729925e-01, 'y': 8.84659079e-02, 'z': 1.17616440e-02, 'yaw': 2.09050032e-01,
                'l': 0, 'w': 0, 'h': 0,
                'dx': 1.17729925e-01, 'dy': 8.84659079e-02, 'dz': 1.17616440e-02, 'dyaw': 2.09050032e-01},
        'car': {'x': 1.58918523e-01, 'y': 1.24935318e-01, 'z': 5.35573165e-03, 'yaw': 9.22800791e-02,
                'l': 0, 'w': 0, 'h': 0,
                'dx': 1.58918523e-01, 'dy': 1.24935318e-01, 'dz': 5.35573165e-03, 'dyaw': 9.22800791e-02},
        'motorcycle': {'x': 3.23647590e-02, 'y': 3.86650974e-02, 'z': 5.47421635e-03, 'yaw': 2.34967407e-01,
                       'l': 0, 'w': 0, 'h': 0,
                       'dx': 3.23647590e-02, 'dy': 3.86650974e-02, 'dz': 5.47421635e-03, 'dyaw': 2.34967407e-01},
        'pedestrian': {'x': 3.34814566e-02, 'y': 2.47354921e-02, 'z': 5.94592529e-03, 'yaw': 4.24962535e-01,
                       'l': 0, 'w': 0, 'h': 0,
                       'dx': 3.34814566e-02, 'dy': 2.47354921e-02, 'dz': 5.94592529e-03, 'dyaw': 4.24962535e-01},
        'trailer': {'x': 4.19985099e-02, 'y': 3.68661552e-02, 'z': 1.19415050e-02, 'yaw': 5.63166240e-02,
                    'l': 0, 'w': 0, 'h': 0,
                    'dx': 4.19985099e-02, 'dy': 3.68661552e-02, 'dz': 1.19415050e-02, 'dyaw': 5.63166240e-02},
        'truck': {'x': 9.45275998e-02, 'y': 9.45620374e-02, 'z': 8.38061721e-03, 'yaw': 1.41680460e-01,
                  'l': 0, 'w': 0, 'h': 0,
                  'dx': 9.45275998e-02, 'dy': 9.45620374e-02, 'dz': 8.38061721e-03, 'dyaw': 1.41680460e-01}
    }

    R = {
        'bicycle': {'x': 0.05390982, 'y': 0.05039431, 'z': 0.01863044, 'yaw': 1.29464435,
                    'l': 0.02713823, 'w': 0.01169572, 'h': 0.01295084},
        'bus': {'x': 0.17546469, 'y': 0.13818929, 'z': 0.05947248, 'yaw': 0.1979503,
                'l': 0.78867322, 'w': 0.05507407, 'h': 0.06684149},
        'car': {'x': 0.08900372, 'y': 0.09412005, 'z': 0.03265469, 'yaw': 1.00535696,
                'l': 0.10912802, 'w': 0.02359175, 'h': 0.02455134},
        'motorcycle': {'x': 0.04052819, 'y':0.0398904, 'z': 0.01511711, 'yaw': 1.06442726,
                       'l': 0.03291016, 'w':0.00957574, 'h': 0.0111605},
        'pedestrian': {'x': 0.03855275, 'y': 0.0377111, 'z': 0.02482115, 'yaw': 2.0751833,
                       'l': 0.02286483, 'w': 0.0136347, 'h': 0.0203149},
        'trailer': {'x': 0.23228021, 'y': 0.22229261, 'z': 0.07006275, 'yaw': 1.05163481,
                    'l': 1.37451601, 'w': 0.06354783, 'h': 0.10500918},
        'truck': {'x': 0.14862173, 'y': 0.1444596, 'z': 0.05417157, 'yaw': 0.73122169,
                  'l': 0.69387238, 'w': 0.05484365, 'h': 0.07748085}
    }

    filter_model = {}

    T = 0.5
    filter_model['F_k'] = np.eye(4, dtype=np.float64)
    I = T*np.eye(2, dtype=np.float64)
    filter_model['F_k'][0:2, 2:4] = I

    filter_model['Q_k']=np.diag([Q[classification]['x'], Q[classification]['y'],Q[classification]['dx'], Q[classification]['dy']])
    filter_model['H_k'] = np.array([[1., 0, 0, 0], [0, 1., 0, 0]], dtype=np.float64)
    filter_model['R_k']=np.diag([R[classification]['x'], R[classification]['y']])
    
    filter_model['p_S'] = p_S
    filter_model['p_D'] = p_D
    P_k = np.diag([P_init**2, P_init**2., 1., 1.])
    filter_model['P_new_birth'] = np.array(P_k, dtype=np.float64)

    filter_model['maximum_number_of_global_hypotheses'] = 1
    filter_model['T_pruning_MBM'] = ber_thr
    filter_model['T_pruning_Pois'] = poi_thr
    filter_model['eB_threshold'] = eB_thr
    filter_model['poission_gating_threshold'] = 4
    filter_model['bernoulli_gating_threshold'] = ber_gating
    filter_model['use_ds_for_pd']=use_ds_as_pd
    filter_model['gating_mode']=gating_mode
 
    x_range = [-50, 50 ]
    y_range = [-50, 50]
    A = (x_range[1] - x_range[0])*(y_range[1]-y_range[0])
    filter_model['clutter_intensity'] = average_number_of_clutter_per_frame/A
    filter_model['xrange'] = x_range
    filter_model['yrange'] = y_range
    filter_model['average_number_of_clutter_per_frame']=average_number_of_clutter_per_frame

    filter_model['state_extraction_option'] = 1
    filter_model['eB_estimation_threshold'] = extraction_thr
    filter_model['use_giou']=use_giou

    return filter_model


def mvnpdf(x, mean, covariance):
    d = mean.shape[0]
    delta_m = x - mean
    pdf_res = 1.0/(np.sqrt((2*np.pi)**d *np.linalg.det(covariance))) * np.exp(-0.5*np.transpose(delta_m).dot(np.linalg.inv(covariance)).dot(delta_m))[0][0]
    return pdf_res

def CardinalityMB(r):
    N = len(r)
    pcard = np.zeros(N+1)
    pcard[1] = 1
    for i in range(N):
        pcard[1:] = (1-r[i])*pcard[1:] + r[i]*pcard[0:i-1]
        pcard[0] = (1-r[i])*pcard[0]

    return pcard
