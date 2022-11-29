from cv2 import normalize
import numpy as np
import copy
import math
from trackers.PMBMGNN.murty import Murty
from trackers.PMBMGNN.util import mvnpdf, CardinalityMB
from functools import reduce
import operator
from scipy.stats import multivariate_normal
from scipy.spatial.distance import mahalanobis as mah
from utils.utils import associate_dets_to_tracks
from utils.utils import giou3d, giou2d

class PMBMGNN_Filter:

    def __init__(self, model): 
        self.model = model

    def predict(self,egoposition,lag_time, filter_pruned,Z_k, birth_rate, noisy_region=1):

  
        F = np.eye(4, dtype=np.float64)
        I = lag_time*np.eye(2, dtype=np.float64)
        F[0:2, 2:4] = I
        Q = self.model['Q_k']
        number_of_surviving_previously_miss_detected_targets = len(filter_pruned['weightPois'])
        
        number_of_surviving_previously_detected_targets=len(filter_pruned['tracks'])
        
        filter_predicted = {}
        filter_predicted['weightPois']=[]
        filter_predicted['meanPois']=[]
        filter_predicted['covPois']=[]
        filter_predicted['rotationPois']=[]
        filter_predicted['elevationPois']=[]
        filter_predicted['sizePois']=[]
        filter_predicted['classificationPois']=[]
        filter_predicted['idPois']=[]
        filter_predicted['detection_scorePois']=[]
        filter_predicted['max_idPois']=filter_pruned['max_idPois']
        
        if number_of_surviving_previously_detected_targets > 0:
            filter_predicted['tracks'] = [{} for i in range(number_of_surviving_previously_detected_targets)]
            filter_predicted['max_idB']=filter_pruned['max_idB']
            filter_predicted['globHyp'] = copy.deepcopy(filter_pruned['globHyp'])
            filter_predicted['globHypWeight'] = copy.deepcopy(filter_pruned['globHypWeight'])
            for previously_detected_target_index in range(number_of_surviving_previously_detected_targets):
                filter_predicted['tracks'][previously_detected_target_index]['eB']=[]
                filter_predicted['tracks'][previously_detected_target_index]['meanB']=[]
                filter_predicted['tracks'][previously_detected_target_index]['covB']=[]
                filter_predicted['tracks'][previously_detected_target_index]['rotationB']=[]
                filter_predicted['tracks'][previously_detected_target_index]['elevationB']=[]
                filter_predicted['tracks'][previously_detected_target_index]['classificationB']=[]
                filter_predicted['tracks'][previously_detected_target_index]['idB']=[]
                filter_predicted['tracks'][previously_detected_target_index]['detection_scoreB']=[]
                
                filter_predicted['tracks'][previously_detected_target_index]['sizeB']=[]
                filter_predicted['tracks'][previously_detected_target_index]['weight_of_single_target_hypothesis_in_log_format']=copy.deepcopy(filter_pruned['tracks'][previously_detected_target_index]['weight_of_single_target_hypothesis_in_log_format'])
                filter_predicted['tracks'][previously_detected_target_index]['giou']=copy.deepcopy(filter_pruned['tracks'][previously_detected_target_index]['giou'])
                filter_predicted['tracks'][previously_detected_target_index]['measurement_association_history']=copy.deepcopy(filter_pruned['tracks'][previously_detected_target_index]['measurement_association_history'])
                
                filter_predicted['tracks'][previously_detected_target_index]['association_counter']=copy.deepcopy(filter_pruned['tracks'][previously_detected_target_index]['association_counter'])

        else:
            filter_predicted['tracks'] = []
            filter_predicted['max_idB']=filter_pruned['max_idB']
            filter_predicted['globHyp'] = []
            filter_predicted['globHypWeight'] = []
        
        if number_of_surviving_previously_miss_detected_targets > 0:
            for PPP_component_index in range(number_of_surviving_previously_miss_detected_targets):
                
                weightPois_previous = filter_pruned['weightPois'][PPP_component_index]
                meanPois_previous = filter_pruned['meanPois'][PPP_component_index]
                detection_scorePois_previous = filter_pruned['detection_scorePois'][PPP_component_index]
                covPois_previous = filter_pruned['covPois'][PPP_component_index]
                rotationPois_previous = filter_pruned['rotationPois'][PPP_component_index]
                elevationPois_previous = filter_pruned['elevationPois'][PPP_component_index]
                classificiationPois_previous = filter_pruned['classificationPois'][PPP_component_index]
                sizePois_previous = filter_pruned['sizePois'][PPP_component_index]
                id_previous=filter_pruned['idPois'][PPP_component_index]
                
                Ps = self.model['p_S']
                meanPois_predicted = F.dot(meanPois_previous)
                distance=math.sqrt((egoposition[0]-meanPois_predicted[0][0])**2+(egoposition[0]-meanPois_predicted[0][0])**2)
                
                if distance > 60:
                    Ps=0.1
                weightPois_predicted = Ps * weightPois_previous
                covPois_predicted = F.dot(covPois_previous).dot(np.transpose(F)+ Q)
                
                filter_predicted['weightPois'].append(weightPois_predicted)       
                filter_predicted['meanPois'].append(meanPois_predicted) 
                filter_predicted['covPois'].append(covPois_predicted)
                filter_predicted['detection_scorePois'].append(detection_scorePois_previous)
                filter_predicted['rotationPois'].append(rotationPois_previous)
                filter_predicted['elevationPois'].append(elevationPois_previous)
                filter_predicted['classificationPois'].append(classificiationPois_previous)
                filter_predicted['sizePois'].append(sizePois_previous)
                filter_predicted['idPois'].append(id_previous)

        number_of_new_birth_targets = len(Z_k)
        trans_width = noisy_region
        for new_birth_target_index in range(number_of_new_birth_targets):
            delta_x = np.random.uniform(-trans_width, trans_width)
            delta_y = np.random.uniform(-trans_width, trans_width)
            
            weightPois_birth = birth_rate
            
            meanPois_birth=np.array([delta_x+Z_k[new_birth_target_index]['translation'][0], delta_y+Z_k[new_birth_target_index]['translation'][1],Z_k[new_birth_target_index]['velocity'][0],Z_k[new_birth_target_index]['velocity'][1]]).reshape(-1,1).astype(np.float64)
            covPois_birth = self.model['P_new_birth']
            
            filter_predicted['weightPois'].append(weightPois_birth) 
            filter_predicted['meanPois'].append(meanPois_birth)
            filter_predicted['covPois'].append(covPois_birth)
            filter_predicted['rotationPois'].append(Z_k[new_birth_target_index]['rotation'])
            filter_predicted['elevationPois'].append(Z_k[new_birth_target_index]['translation'][2])
            filter_predicted['detection_scorePois'].append(Z_k[new_birth_target_index]['detection_score'])
            filter_predicted['classificationPois'].append(Z_k[new_birth_target_index]['detection_name'])
            filter_predicted['sizePois'].append(Z_k[new_birth_target_index]['size'])
            filter_predicted['idPois'].append(filter_predicted['max_idPois']+1+new_birth_target_index)
        filter_predicted['max_idPois']+=number_of_new_birth_targets

        if number_of_surviving_previously_detected_targets > 0:
            for previously_detected_target_index in range(number_of_surviving_previously_detected_targets):
                for single_target_hypothesis_index_from_previous_frame in range(len(filter_pruned['tracks'][previously_detected_target_index]['eB'])):
                    
                    eB_previous = filter_pruned['tracks'][previously_detected_target_index]['eB'][single_target_hypothesis_index_from_previous_frame]
                    meanB_previous = filter_pruned['tracks'][previously_detected_target_index]['meanB'][single_target_hypothesis_index_from_previous_frame]
                    detection_scoreB_previous = filter_pruned['tracks'][previously_detected_target_index]['detection_scoreB'][single_target_hypothesis_index_from_previous_frame]
                    covB_previous = filter_pruned['tracks'][previously_detected_target_index]['covB'][single_target_hypothesis_index_from_previous_frame]
                    rotationB_previous = filter_pruned['tracks'][previously_detected_target_index]['rotationB'][single_target_hypothesis_index_from_previous_frame]       
                    elevationB_previous = filter_pruned['tracks'][previously_detected_target_index]['elevationB'][single_target_hypothesis_index_from_previous_frame]       
                    sizeB_previous = filter_pruned['tracks'][previously_detected_target_index]['sizeB'][single_target_hypothesis_index_from_previous_frame]       
                    classificationB_previous = filter_pruned['tracks'][previously_detected_target_index]['classificationB'][single_target_hypothesis_index_from_previous_frame]
                    idB_previous = filter_pruned['tracks'][previously_detected_target_index]['idB'][single_target_hypothesis_index_from_previous_frame]
                    
                    Ps = self.model['p_S']
                    meanB_predicted = F.dot(meanB_previous)
                    distance=math.sqrt((egoposition[0]-meanB_predicted[0][0])**2+(egoposition[0]-meanB_predicted[0][0])**2)

                    if distance > 60:
                        Ps=0.1
                    eB_predicted = Ps * eB_previous

                    covB_predicted = F.dot(covB_previous).dot(np.transpose(F)) + Q
                                  
                    filter_predicted['tracks'][previously_detected_target_index]['eB'].append(eB_predicted)                    
                    filter_predicted['tracks'][previously_detected_target_index]['meanB'].append(meanB_predicted)
                    filter_predicted['tracks'][previously_detected_target_index]['covB'].append(covB_predicted)
                    filter_predicted['tracks'][previously_detected_target_index]['rotationB'].append(rotationB_previous)
                    filter_predicted['tracks'][previously_detected_target_index]['elevationB'].append(elevationB_previous)
                    filter_predicted['tracks'][previously_detected_target_index]['sizeB'].append(sizeB_previous)
                    filter_predicted['tracks'][previously_detected_target_index]['classificationB'].append(classificationB_previous)
                    filter_predicted['tracks'][previously_detected_target_index]['idB'].append(idB_previous)
                    filter_predicted['tracks'][previously_detected_target_index]['detection_scoreB'].append(detection_scoreB_previous)
                    
        return filter_predicted

    def predict_initial_step(self, Z_k, birth_rate,noisy_region=1):
        filter_predicted = {}
        filter_predicted['weightPois']=[]
        filter_predicted['meanPois']=[]
        filter_predicted['covPois']=[]
        filter_predicted['detection_scorePois']=[]
        filter_predicted['rotationPois']=[]
        filter_predicted['elevationPois']=[]
        filter_predicted['classificationPois']=[]
        filter_predicted['sizePois']=[]
        filter_predicted['tracks'] = []
        filter_predicted['max_idB']=0
        filter_predicted['globHyp'] = []
        filter_predicted['globHypWeight'] = []
        filter_predicted['idPois']=[]
        filter_predicted['max_idPois']=len(Z_k)
    
        number_of_new_birth_targets_init = len(Z_k)

        trans_width = noisy_region

        for new_birth_target_index in range(len(Z_k)):
            delta_x = np.random.uniform(-trans_width, trans_width)
            delta_y = np.random.uniform(-trans_width, trans_width)

            weightPois_birth = birth_rate
            meanPois_birth=np.array([delta_x+Z_k[new_birth_target_index]['translation'][0],delta_y+Z_k[new_birth_target_index]['translation'][1],Z_k[new_birth_target_index]['velocity'][0],Z_k[new_birth_target_index]['velocity'][1]]).reshape(-1,1).astype(np.float64)
            
            covPois_birth = self.model['P_new_birth']
            
            filter_predicted['weightPois'].append(weightPois_birth)
            filter_predicted['meanPois'].append(meanPois_birth)
            filter_predicted['covPois'].append(covPois_birth)
            filter_predicted['rotationPois'].append(Z_k[new_birth_target_index]['rotation'])
            filter_predicted['elevationPois'].append(Z_k[new_birth_target_index]['translation'][2])
            filter_predicted['classificationPois'].append(Z_k[new_birth_target_index]['detection_name'])
            filter_predicted['sizePois'].append(Z_k[new_birth_target_index]['size'])
            filter_predicted['detection_scorePois'].append(Z_k[new_birth_target_index]['detection_score'])
            filter_predicted['idPois'].append(new_birth_target_index)
        return filter_predicted

    def update(self, Z_k, filter_predicted, confidence_score=0, giou_gating=0.15):
        H = self.model['H_k']
        R = self.model['R_k']
        Pd =self.model['p_D']
        po_gating_threshold = self.model['poission_gating_threshold']
        ber_gating_threshold = self.model['bernoulli_gating_threshold']
        clutter_intensity = self.model['clutter_intensity']

        number_of_miss_detected_targets_from_previous_frame_and_new_birth_targets = len(filter_predicted['weightPois'])
        number_of_detected_targets_from_previous_frame = len(filter_predicted['tracks'])
        number_of_global_hypotheses_from_previous_frame = len(filter_predicted['globHyp'])
        number_of_measurements_from_current_frame = len(Z_k)
    
        number_of_previously_undetected_targets_and_new_birth_targets_plus_number_of_clutters = number_of_measurements_from_current_frame
        number_of_potential_detected_targets_at_current_frame_after_update = number_of_detected_targets_from_previous_frame + number_of_previously_undetected_targets_and_new_birth_targets_plus_number_of_clutters

        filter_updated = {}
        filter_updated['weightPois'] = []
        filter_updated['meanPois'] = []
        filter_updated['covPois'] = []
        filter_updated['detection_scorePois'] = []
        filter_updated['rotationPois']=[]
        filter_updated['elevationPois']=[]
        filter_updated['classificationPois']=[]
        filter_updated['sizePois']=[]
        filter_updated['idPois']=[]
        filter_updated['max_idPois']=filter_predicted['max_idPois']

        if number_of_detected_targets_from_previous_frame==0:
            filter_updated['globHyp']=[[int(x) for x in np.zeros(number_of_measurements_from_current_frame)]]
            filter_updated['globHypWeight']=[1]
            if number_of_measurements_from_current_frame == 0:
                filter_updated['tracks'] = []
                filter_updated['max_idB']=0            
            else: 
                filter_updated['tracks']=[{} for n in range(number_of_measurements_from_current_frame)]
                for i in range(number_of_measurements_from_current_frame):
                    filter_updated['tracks'][i]['eB']= []
                    filter_updated['tracks'][i]['covB']= []
                    filter_updated['tracks'][i]['meanB']= []
                    filter_updated['tracks'][i]['rotationB']=[]
                    filter_updated['tracks'][i]['elevationB']=[]
                    filter_updated['tracks'][i]['classificationB']=[]
                    filter_updated['tracks'][i]['idB']=[]
                    filter_updated['tracks'][i]['detection_scoreB']=[]
                    filter_updated['tracks'][i]['sizeB']=[]
                    filter_updated['tracks'][i]['weight_of_single_target_hypothesis_in_log_format']= []
                    filter_updated['tracks'][i]['giou']= []
                    filter_updated['tracks'][i]['single_target_hypothesis_index_from_previous_frame']=[]
                    filter_updated['tracks'][i]['measurement_association_history']= []
                    filter_updated['tracks'][i]['measurement_association_from_this_frame']= []
                    filter_updated['tracks'][i]['association_counter']= []
            filter_updated['max_idB']=filter_predicted['max_idB']+number_of_measurements_from_current_frame
        else:
            filter_updated['globHyp'] = []
            filter_updated['globHypWeight'] = []
            filter_updated['tracks']=[{} for n in range(number_of_detected_targets_from_previous_frame+number_of_measurements_from_current_frame)]
            
            for previously_detected_target_index in range(number_of_detected_targets_from_previous_frame):
                number_of_single_target_hypotheses_from_previous_frame = len(filter_predicted['tracks'][previously_detected_target_index]['eB'])
                filter_updated['tracks'][previously_detected_target_index]['eB'] = []
                filter_updated['tracks'][previously_detected_target_index]['meanB'] = []
                filter_updated['tracks'][previously_detected_target_index]['covB'] = []
                filter_updated['tracks'][previously_detected_target_index]['idB'] = []
                filter_updated['tracks'][previously_detected_target_index]['detection_scoreB'] = []
                
                filter_updated['tracks'][previously_detected_target_index]['rotationB'] = []
                filter_updated['tracks'][previously_detected_target_index]['elevationB'] = []
                filter_updated['tracks'][previously_detected_target_index]['classificationB'] = []
                filter_updated['tracks'][previously_detected_target_index]['sizeB'] = []
                filter_updated['tracks'][previously_detected_target_index]['weight_of_single_target_hypothesis_in_log_format'] = []
                filter_updated['tracks'][previously_detected_target_index]['giou'] = []
                filter_updated['tracks'][previously_detected_target_index]['single_target_hypothesis_index_from_previous_frame']=[]
                filter_updated['tracks'][previously_detected_target_index]['measurement_association_history'] = copy.deepcopy(filter_predicted['tracks'][previously_detected_target_index]['measurement_association_history'])
                filter_updated['tracks'][previously_detected_target_index]['measurement_association_from_this_frame'] = []
                filter_updated['tracks'][previously_detected_target_index]['association_counter'] = []

            for i in range(number_of_measurements_from_current_frame):
                filter_updated['tracks'][number_of_detected_targets_from_previous_frame + i]['eB']= []
                filter_updated['tracks'][number_of_detected_targets_from_previous_frame + i]['meanB']= []
                filter_updated['tracks'][number_of_detected_targets_from_previous_frame + i]['covB']= []
                filter_updated['tracks'][number_of_detected_targets_from_previous_frame + i]['idB']=[]
                filter_updated['tracks'][number_of_detected_targets_from_previous_frame + i]['detection_scoreB']=[]
                filter_updated['tracks'][number_of_detected_targets_from_previous_frame + i]['rotationB']= []
                filter_updated['tracks'][number_of_detected_targets_from_previous_frame + i]['elevationB']= []
                filter_updated['tracks'][number_of_detected_targets_from_previous_frame + i]['classificationB']= []
                filter_updated['tracks'][number_of_detected_targets_from_previous_frame + i]['sizeB']= []
                filter_updated['tracks'][number_of_detected_targets_from_previous_frame + i]['weight_of_single_target_hypothesis_in_log_format']= []
                filter_updated['tracks'][number_of_detected_targets_from_previous_frame + i]['giou']= []
                filter_updated['tracks'][number_of_detected_targets_from_previous_frame + i]['single_target_hypothesis_index_from_previous_frame']=[]
                filter_updated['tracks'][number_of_detected_targets_from_previous_frame + i]['measurement_association_history']= []
                filter_updated['tracks'][number_of_detected_targets_from_previous_frame + i]['measurement_association_from_this_frame']= []
                filter_updated['tracks'][number_of_detected_targets_from_previous_frame + i]['association_counter']= []

            filter_updated['max_idB']=filter_predicted['max_idB']+number_of_measurements_from_current_frame

        for PPP_component_index in range(number_of_miss_detected_targets_from_previous_frame_and_new_birth_targets):
            weightPois_predicted = filter_predicted['weightPois'][PPP_component_index]
            meanPois_predicted = filter_predicted['meanPois'][PPP_component_index]
            covPois_predicted = filter_predicted['covPois'][PPP_component_index]
            detection_scorePois_predicted = filter_predicted['detection_scorePois'][PPP_component_index]
            rotationPois_predicted = filter_predicted['rotationPois'][PPP_component_index]
            elevationPois_predicted = filter_predicted['elevationPois'][PPP_component_index]
            classificationPois_predicted = filter_predicted['classificationPois'][PPP_component_index]
            sizePois_predicted = filter_predicted['sizePois'][PPP_component_index]
            idPois_predicted = filter_predicted['idPois'][PPP_component_index]

            if self.model['use_ds_for_pd']:
                wegithPois_updated = (1-filter_predicted['detection_scorePois'][PPP_component_index]) * weightPois_predicted
            else:
                wegithPois_updated = (1-Pd) * weightPois_predicted
            
            meanPois_updated = meanPois_predicted
            covPois_updated = covPois_predicted
            
            filter_updated['weightPois'].append(wegithPois_updated)
            filter_updated['meanPois'].append(meanPois_updated)
            filter_updated['covPois'].append(covPois_updated)
            filter_updated['rotationPois'].append(rotationPois_predicted)
            filter_updated['elevationPois'].append(elevationPois_predicted)
            filter_updated['detection_scorePois'].append(detection_scorePois_predicted)
            filter_updated['classificationPois'].append(classificationPois_predicted)
            filter_updated['sizePois'].append(sizePois_predicted)
            filter_updated['idPois'].append(idPois_predicted)
        filter_updated['max_idPois']=filter_predicted['max_idPois']

        for measurement_index in range(number_of_measurements_from_current_frame):    
            tracks_associated_with_this_measurement = []
        
            for PPP_component_index in range(number_of_miss_detected_targets_from_previous_frame_and_new_birth_targets):
                mean_PPP_component_predicted = filter_predicted['meanPois'][PPP_component_index]
                cov_PPP_component_predicted = filter_predicted['covPois'][PPP_component_index]

                mean_PPP_component_measured = H.dot(mean_PPP_component_predicted).astype('float64')             
                S_PPP_component = (H.dot(cov_PPP_component_predicted).dot(np.transpose(H))+R).astype('float64')

                S_PPP_component = 0.5 * (S_PPP_component + np.transpose(S_PPP_component))
                
                ppp_innovation_residual = np.array([Z_k[measurement_index]['translation'][0] - mean_PPP_component_measured[0],Z_k[measurement_index]['translation'][1] - mean_PPP_component_measured[1]]).reshape(-1,1).astype(np.float64)
                Si = copy.deepcopy(S_PPP_component)
                invSi = np.linalg.inv(np.array(Si, dtype=np.float64))
                
                track={}
                temp=mean_PPP_component_predicted.reshape(-1,1).tolist()
                track['translation']=[temp[0][0],temp[1][0]]
                track['translation'].append(elevationPois_predicted)
                track['rotation']=rotationPois_predicted
                track['size']=sizePois_predicted

                if self.model['gating_mode']=='mahalanobis':
                    maha1 = np.transpose(ppp_innovation_residual).dot(invSi).dot(ppp_innovation_residual)[0][0]
                    value=maha1
                    gating_threshold=ber_gating_threshold
                    if value < gating_threshold: 
                        tracks_associated_with_this_measurement.append(PPP_component_index)                
                elif self.model['gating_mode']=='giou':
                    value = giou3d(Z_k[measurement_index],track)
                    gating_threshold=giou_gating
                    if value >= gating_threshold: 
                        tracks_associated_with_this_measurement.append(PPP_component_index)  

            if len(tracks_associated_with_this_measurement)>0:
                meanB_sum = np.zeros((len(H[0]),1))
                covB_sum = np.zeros((len(H[0]),len(H[0])))
                weight_of_true_detection = 0
                for associated_track_index in tracks_associated_with_this_measurement:
                    
                    mean_associated_track_predicted = filter_predicted['meanPois'][associated_track_index]
                    cov_associated_track_predicted = filter_predicted['covPois'][associated_track_index]
                    weight_associated_track_predicted = filter_predicted['weightPois'][associated_track_index]
                    
                    mean_associated_track_measured = H.dot(mean_associated_track_predicted).astype('float64')                     
                    
                    S_associated_track = (H.dot(cov_associated_track_predicted).dot(np.transpose(H))+R).astype('float64')
                    
                    S_associated_track = 0.5 * (S_associated_track + np.transpose(S_associated_track))

                    Vs= np.linalg.cholesky(S_associated_track) 
                    Vs = np.matrix(Vs)
                    
                    Si = copy.deepcopy(Vs)
                    inv_sqrt_Si = np.linalg.inv(np.array(Si, dtype=np.float64))
                    invSi= inv_sqrt_Si*np.transpose(inv_sqrt_Si)

                    K_associated_track = cov_associated_track_predicted.dot(np.transpose(H)).dot(invSi).astype('float64')
                    track_innovation_residual = np.array([Z_k[measurement_index]['translation'][0] - mean_associated_track_measured[0],Z_k[measurement_index]['translation'][1] - mean_associated_track_measured[1]]).reshape(-1,1).astype(np.float64)
                    
                    mean_associated_track_updated = mean_associated_track_predicted + K_associated_track.dot(track_innovation_residual)

                    cov_associated_track_updated = cov_associated_track_predicted - K_associated_track.dot(H).dot(cov_associated_track_predicted).astype('float64')
                    
                    cov_associated_track_updated = 0.5 * (cov_associated_track_updated + np.transpose(cov_associated_track_updated))               
                    mvnpdf_value=mvnpdf(np.array([[Z_k[measurement_index]['translation'][0]],[Z_k[measurement_index]['translation'][1]]]), np.array([mean_associated_track_measured[0],mean_associated_track_measured[1]]),S_associated_track)
                    if self.model['use_ds_for_pd']:
                        weight_for_track_detection = Z_k[measurement_index]['detection_score']*weight_associated_track_predicted*mvnpdf(np.array([[Z_k[measurement_index]['translation'][0]],[Z_k[measurement_index]['translation'][1]]]), np.array([mean_associated_track_measured[0],mean_associated_track_measured[1]]),S_associated_track)
                    else:
                        weight_for_track_detection = Pd*weight_associated_track_predicted*mvnpdf_value
                    
                    weight_of_true_detection += weight_for_track_detection
                    meanB_sum += weight_for_track_detection*(mean_associated_track_updated)
                    covB_sum += weight_for_track_detection*cov_associated_track_updated + weight_for_track_detection*(mean_associated_track_updated.dot(np.transpose(mean_associated_track_updated)))


                meanB_updated=meanB_sum/weight_of_true_detection
                covB_updated = covB_sum/weight_of_true_detection - (meanB_updated*np.transpose(meanB_updated))
                rotationB_updated = Z_k[measurement_index]['rotation']
                elevationB_updated = Z_k[measurement_index]['translation'][2]
                classificationB_updated = Z_k[measurement_index]['detection_name']
                sizeB_updated = Z_k[measurement_index]['size']

                probability_of_detection = weight_of_true_detection + clutter_intensity
                if Z_k[measurement_index]['detection_score'] > confidence_score:
                    eB_updated = 1
                else:
                    eB_updated = weight_of_true_detection/probability_of_detection
                
                filter_updated['tracks'][number_of_detected_targets_from_previous_frame + measurement_index]['detection_scoreB'].append(Z_k[measurement_index]['detection_score'])
                filter_updated['tracks'][number_of_detected_targets_from_previous_frame + measurement_index]['eB'].append(eB_updated)
                filter_updated['tracks'][number_of_detected_targets_from_previous_frame + measurement_index]['meanB'].append(meanB_updated)
                filter_updated['tracks'][number_of_detected_targets_from_previous_frame + measurement_index]['covB'].append(covB_updated)
                filter_updated['tracks'][number_of_detected_targets_from_previous_frame + measurement_index]['rotationB'].append(rotationB_updated)
                filter_updated['tracks'][number_of_detected_targets_from_previous_frame + measurement_index]['elevationB'].append(elevationB_updated)
                filter_updated['tracks'][number_of_detected_targets_from_previous_frame + measurement_index]['classificationB'].append(classificationB_updated)
                filter_updated['tracks'][number_of_detected_targets_from_previous_frame + measurement_index]['idB'].append(filter_predicted['max_idB']+measurement_index+1)
                filter_updated['tracks'][number_of_detected_targets_from_previous_frame + measurement_index]['sizeB'].append(sizeB_updated)
                filter_updated['tracks'][number_of_detected_targets_from_previous_frame + measurement_index]['weight_of_single_target_hypothesis_in_log_format'].append(np.log(probability_of_detection))
                filter_updated['tracks'][number_of_detected_targets_from_previous_frame + measurement_index]['giou'].append(probability_of_detection)

                filter_updated['tracks'][number_of_detected_targets_from_previous_frame + measurement_index]['single_target_hypothesis_index_from_previous_frame'].append(-1)
                filter_updated['tracks'][number_of_detected_targets_from_previous_frame + measurement_index]['measurement_association_history'].append(measurement_index)
                filter_updated['tracks'][number_of_detected_targets_from_previous_frame + measurement_index]['measurement_association_from_this_frame'].append(measurement_index)
                filter_updated['tracks'][number_of_detected_targets_from_previous_frame + measurement_index]['association_counter'].append(1)

            else: 
                meanB_updated = mean_PPP_component_predicted
                covB_updated = cov_PPP_component_predicted
                rotationB_updated = Z_k[measurement_index]['rotation']
                elevationB_updated = Z_k[measurement_index]['translation'][2]
                sizeB_updated = Z_k[measurement_index]['size']
                classificationB_updated=Z_k[measurement_index]['detection_name']

                probability_of_detection = clutter_intensity
                eB_updated = 0


                filter_updated['tracks'][number_of_detected_targets_from_previous_frame + measurement_index]['detection_scoreB'].append(Z_k[measurement_index]['detection_score'])
                filter_updated['tracks'][number_of_detected_targets_from_previous_frame + measurement_index]['eB'].append(eB_updated)
                filter_updated['tracks'][number_of_detected_targets_from_previous_frame + measurement_index]['weight_of_single_target_hypothesis_in_log_format'].append(np.log(probability_of_detection))
                filter_updated['tracks'][number_of_detected_targets_from_previous_frame + measurement_index]['giou'].append(np.log(0.15))

                filter_updated['tracks'][number_of_detected_targets_from_previous_frame + measurement_index]['meanB'].append(meanB_updated)
                filter_updated['tracks'][number_of_detected_targets_from_previous_frame + measurement_index]['covB'].append(covB_updated)
                filter_updated['tracks'][number_of_detected_targets_from_previous_frame + measurement_index]['rotationB'].append(rotationB_updated)
                filter_updated['tracks'][number_of_detected_targets_from_previous_frame + measurement_index]['elevationB'].append(elevationB_updated)
                filter_updated['tracks'][number_of_detected_targets_from_previous_frame + measurement_index]['sizeB'].append(sizeB_updated)
                filter_updated['tracks'][number_of_detected_targets_from_previous_frame + measurement_index]['idB'].append(filter_predicted['max_idB']+measurement_index+1)
                filter_updated['tracks'][number_of_detected_targets_from_previous_frame + measurement_index]['classificationB'].append(classificationB_updated)
                filter_updated['tracks'][number_of_detected_targets_from_previous_frame + measurement_index]['single_target_hypothesis_index_from_previous_frame'].append(-1)
                filter_updated['tracks'][number_of_detected_targets_from_previous_frame + measurement_index]['measurement_association_history'].append(-1)
                filter_updated['tracks'][number_of_detected_targets_from_previous_frame + measurement_index]['measurement_association_from_this_frame'].append(-1)
                filter_updated['tracks'][number_of_detected_targets_from_previous_frame + measurement_index]['measurement_association_from_this_frame'].append(0)
       
        for previously_detected_target_index in range(number_of_detected_targets_from_previous_frame):
            
            number_of_single_target_hypotheses_from_previous_frame = len(filter_predicted['tracks'][previously_detected_target_index]['eB'])

            for single_target_hypothesis_index_from_previous_frame in range(number_of_single_target_hypotheses_from_previous_frame):
                
                mean_single_target_hypothesis_predicted = filter_predicted['tracks'][previously_detected_target_index]['meanB'][single_target_hypothesis_index_from_previous_frame]
                cov_single_target_hypothesis_predicted = filter_predicted['tracks'][previously_detected_target_index]['covB'][single_target_hypothesis_index_from_previous_frame]
                eB_single_target_hypothesis_predicted = filter_predicted['tracks'][previously_detected_target_index]['eB'][single_target_hypothesis_index_from_previous_frame]
                rotationB_single_target_hypothesis_predicted = filter_predicted['tracks'][previously_detected_target_index]['rotationB'][single_target_hypothesis_index_from_previous_frame]
                elevationB_single_target_hypothesis_predicted = filter_predicted['tracks'][previously_detected_target_index]['elevationB'][single_target_hypothesis_index_from_previous_frame]
                classificationB_single_target_hypothesis_predicted = filter_predicted['tracks'][previously_detected_target_index]['classificationB'][single_target_hypothesis_index_from_previous_frame]
                sizeB_single_target_hypothesis_predicted = filter_predicted['tracks'][previously_detected_target_index]['sizeB'][single_target_hypothesis_index_from_previous_frame]
                idB_single_target_hypothesis_predicted = filter_predicted['tracks'][previously_detected_target_index]['idB'][single_target_hypothesis_index_from_previous_frame]
                association_counter_before = filter_predicted['tracks'][previously_detected_target_index]['association_counter'][single_target_hypothesis_index_from_previous_frame]
                
                detection_scoreB_single_target_hypothesis_predicted = filter_predicted['tracks'][previously_detected_target_index]['detection_scoreB'][single_target_hypothesis_index_from_previous_frame]

                track={}
                temp=mean_single_target_hypothesis_predicted.reshape(-1,1).tolist()
                track['translation']=[temp[0][0],temp[1][0]]
                track['translation'].append(elevationB_single_target_hypothesis_predicted)
                track['rotation']=rotationB_single_target_hypothesis_predicted
                track['size']=sizeB_single_target_hypothesis_predicted

                if self.model['use_ds_for_pd']:
                    probability_for_track_exist_but_undetected = eB_single_target_hypothesis_predicted*(1-detection_scoreB_single_target_hypothesis_predicted)
                else:
                    probability_for_track_exist_but_undetected = eB_single_target_hypothesis_predicted*(1-Pd)
                probability_for_track_dose_not_exit = 1-eB_single_target_hypothesis_predicted
                eB_undetected = probability_for_track_exist_but_undetected/(probability_for_track_exist_but_undetected+probability_for_track_dose_not_exit) 
                weight_of_single_target_hypothesis_undetected_in_log_format = np.log(probability_for_track_exist_but_undetected+probability_for_track_dose_not_exit)
                mean_single_target_hypothesis_undetected = mean_single_target_hypothesis_predicted
                cov_single_target_hypothesis_undetected = cov_single_target_hypothesis_predicted
                
                filter_updated['tracks'][previously_detected_target_index]['meanB'].append(mean_single_target_hypothesis_undetected)
                filter_updated['tracks'][previously_detected_target_index]['covB'].append(cov_single_target_hypothesis_undetected)
                filter_updated['tracks'][previously_detected_target_index]['rotationB'].append(rotationB_single_target_hypothesis_predicted)
                filter_updated['tracks'][previously_detected_target_index]['elevationB'].append(elevationB_single_target_hypothesis_predicted)
                filter_updated['tracks'][previously_detected_target_index]['classificationB'].append(classificationB_single_target_hypothesis_predicted)
                filter_updated['tracks'][previously_detected_target_index]['idB'].append(idB_single_target_hypothesis_predicted)
                
                filter_updated['tracks'][previously_detected_target_index]['sizeB'].append(sizeB_single_target_hypothesis_predicted)
                filter_updated['tracks'][previously_detected_target_index]['eB'].append(eB_undetected)
                filter_updated['tracks'][previously_detected_target_index]['detection_scoreB'].append(detection_scoreB_single_target_hypothesis_predicted)
                filter_updated['tracks'][previously_detected_target_index]['weight_of_single_target_hypothesis_in_log_format'].append(weight_of_single_target_hypothesis_undetected_in_log_format)
                filter_updated['tracks'][previously_detected_target_index]['giou'].append(0)

                filter_updated['tracks'][previously_detected_target_index]['single_target_hypothesis_index_from_previous_frame'].append(single_target_hypothesis_index_from_previous_frame)
                filter_updated['tracks'][previously_detected_target_index]['measurement_association_history'].append(-1)
                filter_updated['tracks'][previously_detected_target_index]['measurement_association_from_this_frame'].append(-1)
                filter_updated['tracks'][previously_detected_target_index]['association_counter'].append(association_counter_before)
                
                S_single_target_hypothesis = (H.dot(cov_single_target_hypothesis_predicted).dot(np.transpose(H))+R).astype('float64')
                
                S_single_target_hypothesis = 0.5 * (S_single_target_hypothesis + np.transpose(S_single_target_hypothesis))
                mean_single_target_hypothesis_measured = H.dot(mean_single_target_hypothesis_predicted).astype('float64')
                
                Si = copy.deepcopy(S_single_target_hypothesis)
                invSi = np.linalg.inv(np.array(Si, dtype=np.float64))

                K_single_target_hypothesis = cov_single_target_hypothesis_predicted.dot(np.transpose(H)).dot(invSi).astype('float64')
                starting_position_idx = len(filter_updated['tracks'][previously_detected_target_index]['meanB'])
                associated_measurement_counter = 0          
                for measurement_index in range(number_of_measurements_from_current_frame):
                    detected_track_innovation_residual = np.array([Z_k[measurement_index]['translation'][0] - mean_single_target_hypothesis_measured[0],Z_k[measurement_index]['translation'][1] - mean_single_target_hypothesis_measured[1]]).reshape(-1,1).astype(np.float64)
                    if self.model['gating_mode']=='mahalanobis':
                        maha2 = np.transpose(detected_track_innovation_residual).dot(invSi).dot(detected_track_innovation_residual)[0][0]
                        value=maha2
                        gating_threshold=ber_gating_threshold
                        if value < gating_threshold:
                            within_gating = True
                        else:
                            within_gating = False
                    elif self.model['gating_mode']=='giou':
                        value = giou3d(Z_k[measurement_index],track)
                        gating_threshold=giou_gating
                        if value >= gating_threshold:
                            within_gating = True
                        else:
                            within_gating = False

                    if within_gating: 
                        associated_measurement_counter += 1
                        mean_single_target_hypothesis_updated = mean_single_target_hypothesis_predicted + K_single_target_hypothesis.dot(detected_track_innovation_residual) # it is a column vector with lenghth 4
                        cov_single_target_hypothesis_updated = cov_single_target_hypothesis_predicted - K_single_target_hypothesis.dot(H).dot(cov_single_target_hypothesis_predicted).astype('float64')

                        cov_single_target_hypothesis_updated = 0.5 * (cov_single_target_hypothesis_updated + np.transpose(cov_single_target_hypothesis_updated))
                        mvnpdf_value = mvnpdf(np.array([[Z_k[measurement_index]['translation'][0]],[Z_k[measurement_index]['translation'][1]]]), np.array([mean_single_target_hypothesis_measured[0],mean_single_target_hypothesis_measured[1]]),S_single_target_hypothesis)
                        if self.model['use_ds_for_pd']:
                            weight_of_single_target_hypothesis_updated_in_log_format =np.log(Z_k[measurement_index]['detection_score'] * eB_single_target_hypothesis_predicted * mvnpdf(np.array([[Z_k[measurement_index]['translation'][0]],[Z_k[measurement_index]['translation'][1]]]), np.array([mean_single_target_hypothesis_measured[0],mean_single_target_hypothesis_measured[1]]),S_single_target_hypothesis))
                        else:
                            weight_of_single_target_hypothesis_updated_in_log_format =np.log(Pd * eB_single_target_hypothesis_predicted * mvnpdf(np.array([[Z_k[measurement_index]['translation'][0]],[Z_k[measurement_index]['translation'][1]]]), np.array([mean_single_target_hypothesis_measured[0],mean_single_target_hypothesis_measured[1]]),S_single_target_hypothesis))

                        if self.model['use_giou']:
                            giou_value = giou3d(Z_k[measurement_index],track) 
                            if giou_value <= 0:
                                print('giou smaller than 0')
                                giou=-50
                            else:
                                giou=np.log(giou_value)
                        else:
                            giou=0
                        
                        eB_single_target_hypothesis_updated = 1
                        rotationB_single_target_hypothesis_updated = Z_k[measurement_index]['rotation']
                        elevationB_single_target_hypothesis_updated = Z_k[measurement_index]['translation'][2]
                        classificationB_single_target_hypothesis_updated = Z_k[measurement_index]['detection_name']
                        sizeB_single_target_hypothesis_updated = Z_k[measurement_index]['size']
                        idB_single_target_hypothesis_updated = filter_predicted['tracks'][previously_detected_target_index]['idB'][single_target_hypothesis_index_from_previous_frame]
                        filter_updated['tracks'][previously_detected_target_index]['meanB'].append(mean_single_target_hypothesis_updated)
                        filter_updated['tracks'][previously_detected_target_index]['covB'].append(cov_single_target_hypothesis_updated)
                        filter_updated['tracks'][previously_detected_target_index]['rotationB'].append(rotationB_single_target_hypothesis_updated)
                        filter_updated['tracks'][previously_detected_target_index]['elevationB'].append(elevationB_single_target_hypothesis_updated)
                        filter_updated['tracks'][previously_detected_target_index]['classificationB'].append(classificationB_single_target_hypothesis_updated)
                        filter_updated['tracks'][previously_detected_target_index]['sizeB'].append(sizeB_single_target_hypothesis_updated)
                        filter_updated['tracks'][previously_detected_target_index]['eB'].append(eB_single_target_hypothesis_updated)
                        filter_updated['tracks'][previously_detected_target_index]['idB'].append(idB_single_target_hypothesis_updated)
                        filter_updated['tracks'][previously_detected_target_index]['detection_scoreB'].append(Z_k[measurement_index]['detection_score'])
                        filter_updated['tracks'][previously_detected_target_index]['weight_of_single_target_hypothesis_in_log_format'].append(weight_of_single_target_hypothesis_updated_in_log_format)
                        filter_updated['tracks'][previously_detected_target_index]['giou'].append(giou)

                        filter_updated['tracks'][previously_detected_target_index]['single_target_hypothesis_index_from_previous_frame'].append(single_target_hypothesis_index_from_previous_frame)
                        filter_updated['tracks'][previously_detected_target_index]['measurement_association_history'].append(measurement_index)
                        filter_updated['tracks'][previously_detected_target_index]['measurement_association_from_this_frame'].append(measurement_index)
                        filter_updated['tracks'][previously_detected_target_index]['association_counter'].append(association_counter_before+1)

        if number_of_measurements_from_current_frame == 0:
            filter_updated['globHyp']= [[int(x) for x in np.zeros(number_of_detected_targets_from_previous_frame)]]
            filter_updated['globHypWeight']=[1]
        else:
            if number_of_detected_targets_from_previous_frame>0:
    
                weight_of_global_hypothesis_in_log_format=[]
                globHyp=[]
    
                for global_hypothesis_index_from_pevious_frame in range(number_of_global_hypotheses_from_previous_frame):
                    cost_matrix_log=-np.inf*np.ones((number_of_detected_targets_from_previous_frame+number_of_measurements_from_current_frame , number_of_measurements_from_current_frame))
                    weight_for_missed_detection_hypotheses=np.zeros(number_of_detected_targets_from_previous_frame)
                    optimal_associations_all = []
                    for previously_detected_target_index in range(number_of_detected_targets_from_previous_frame):
                        single_target_hypothesis_index_specified_by_previous_step_global_hypothesis=filter_predicted['globHyp'][global_hypothesis_index_from_pevious_frame][previously_detected_target_index]
                        if single_target_hypothesis_index_specified_by_previous_step_global_hypothesis != -1: # if this track exist                                
                            new_single_target_hypotheses_indices_generated_under_this_previous_global_hypothesis = [idx for idx, value in enumerate(filter_updated['tracks'][previously_detected_target_index]['single_target_hypothesis_index_from_previous_frame']) if value == single_target_hypothesis_index_specified_by_previous_step_global_hypothesis]
                            missed_detection_hypothesis_weight = filter_updated['tracks'][previously_detected_target_index]['weight_of_single_target_hypothesis_in_log_format'][new_single_target_hypotheses_indices_generated_under_this_previous_global_hypothesis[0]]
                            weight_for_missed_detection_hypotheses[previously_detected_target_index]=missed_detection_hypothesis_weight
                            measurement_association_list_generated_under_this_previous_global_hypothesis = [filter_updated['tracks'][previously_detected_target_index]['measurement_association_from_this_frame'][x] for x in new_single_target_hypotheses_indices_generated_under_this_previous_global_hypothesis[1:]]
                            if len(measurement_association_list_generated_under_this_previous_global_hypothesis) >0:
                                for idx, associated_measurement in enumerate(measurement_association_list_generated_under_this_previous_global_hypothesis):
                                    idx_of_current_single_target_hypothesis = new_single_target_hypotheses_indices_generated_under_this_previous_global_hypothesis[idx+1]
                                    if self.model['use_giou']:
                                        cost_matrix_log[previously_detected_target_index][associated_measurement] = \
                                        filter_updated['tracks'][previously_detected_target_index]['weight_of_single_target_hypothesis_in_log_format'][idx_of_current_single_target_hypothesis]\
                                        -missed_detection_hypothesis_weight+filter_updated['tracks'][previously_detected_target_index]['giou'][idx_of_current_single_target_hypothesis]
                                    else:
                                        cost_matrix_log[previously_detected_target_index][associated_measurement] = \
                                        filter_updated['tracks'][previously_detected_target_index]['weight_of_single_target_hypothesis_in_log_format'][idx_of_current_single_target_hypothesis]\
                                        -missed_detection_hypothesis_weight
        
                    for measurement_index in range(number_of_measurements_from_current_frame):
                        if self.model['use_giou']:
                            cost_matrix_log[number_of_detected_targets_from_previous_frame+measurement_index][measurement_index]=np.log(0.15)+filter_updated['tracks'][number_of_detected_targets_from_previous_frame+measurement_index]['weight_of_single_target_hypothesis_in_log_format'][0]
                        else:
                            cost_matrix_log[number_of_detected_targets_from_previous_frame+measurement_index][measurement_index]=filter_updated['tracks'][number_of_detected_targets_from_previous_frame+measurement_index]['weight_of_single_target_hypothesis_in_log_format'][0]

                    indices_of_cost_matrix_with_valid_elements = 1 - np.isinf(cost_matrix_log)
                    indices_of_measurements_non_exclusive = [x for x in range(len(indices_of_cost_matrix_with_valid_elements[0])) if sum(indices_of_cost_matrix_with_valid_elements[:,x])>1]
                    if len(indices_of_measurements_non_exclusive)>0:
                        indices_of_tracks_non_exclusive=[x for x in range(len(np.transpose([indices_of_cost_matrix_with_valid_elements[:,x] for x in indices_of_measurements_non_exclusive]))) if sum(np.transpose([indices_of_cost_matrix_with_valid_elements[:,x] for x in indices_of_measurements_non_exclusive])[x]>0)]
                        cost_matrix_log_non_exclusive = np.array(np.transpose([cost_matrix_log[:,x] for x in indices_of_measurements_non_exclusive]))[indices_of_tracks_non_exclusive]
                    else:
                        indices_of_tracks_non_exclusive = []
                        cost_matrix_log_non_exclusive = []
                    
                    indices_of_measurements_exclusive = [x for x in range(len(indices_of_cost_matrix_with_valid_elements[0])) if sum(indices_of_cost_matrix_with_valid_elements[:,x])==1]
                    if len(indices_of_measurements_exclusive) > 0:
                        indices_of_tracks_exclusive = [np.argmax(indices_of_cost_matrix_with_valid_elements[:,x]) for x in indices_of_measurements_exclusive]
                        cost_matrix_log_exclusive = np.array(np.transpose([cost_matrix_log[:,x] for x in indices_of_measurements_exclusive]))[indices_of_tracks_exclusive]
                    else:
                        indices_of_tracks_exclusive = []
                        cost_matrix_log_exclusive = []
                    
                    if len(cost_matrix_log_non_exclusive)==0:
                        association_vector=np.zeros(number_of_measurements_from_current_frame)
                        for index_of_idx, idx in enumerate(indices_of_measurements_exclusive):
                            association_vector[idx]=indices_of_tracks_exclusive[index_of_idx]
                        optimal_associations_all.append(association_vector)
                        cost_for_optimal_associations_non_exclusive = [0]
    
                    else:
                        k_best_global_hypotheses_under_this_previous_global_hypothesis=np.ceil(self.model['maximum_number_of_global_hypotheses']*filter_predicted['globHypWeight'][global_hypothesis_index_from_pevious_frame])
                        cost_matrix_object_went_though_murty = Murty(-np.transpose(cost_matrix_log_non_exclusive)) 
                        optimal_associations_non_exclusive = []
                        cost_for_optimal_associations_non_exclusive = []
                        for iterate in range(int(k_best_global_hypotheses_under_this_previous_global_hypothesis)):
                            still_valid_flag,ith_optimal_cost,ith_optimal_solution = cost_matrix_object_went_though_murty.draw()
                            if still_valid_flag == True:
                                optimal_associations_non_exclusive.append(ith_optimal_solution)
                                cost_for_optimal_associations_non_exclusive.append(ith_optimal_cost)
                            else:
                                break

                        optimal_associations_all = -np.inf*np.ones((len(optimal_associations_non_exclusive),number_of_measurements_from_current_frame))           
                        for ith_optimal_option_index, ith_optimal_association_vector in enumerate(optimal_associations_non_exclusive):

                            for idx_of_non_exclusive_matrix, ith_optimal_track_idx in enumerate(ith_optimal_association_vector):

                                actual_measurement_idx = indices_of_measurements_non_exclusive[idx_of_non_exclusive_matrix]
                                actual_track_idx = indices_of_tracks_non_exclusive[ith_optimal_track_idx]
                                optimal_associations_all[ith_optimal_option_index][actual_measurement_idx]=actual_track_idx

                            for idx_of_exclusive_matrix, actual_measurement_idx in enumerate(indices_of_measurements_exclusive):
                                actual_track_idx = indices_of_tracks_exclusive[idx_of_exclusive_matrix]
                                optimal_associations_all[ith_optimal_option_index][actual_measurement_idx]= actual_track_idx

                    weight_of_exclusive_assosications = 0
                    for row_index in range(len(cost_matrix_log_exclusive)):
                        weight_of_exclusive_assosications += cost_matrix_log_exclusive[row_index][row_index]

                    for ith_optimal_option in range(len(optimal_associations_all)):
                        weight_of_global_hypothesis_in_log_format.append(-cost_for_optimal_associations_non_exclusive[ith_optimal_option]+np.sum(weight_for_missed_detection_hypotheses)+weight_of_exclusive_assosications+np.log(filter_predicted['globHypWeight'][global_hypothesis_index_from_pevious_frame])) # The global weight associated with this hypothesis
    
                    globHyp_from_current_frame_under_this_globHyp_from_previous_frame=np.zeros((len(optimal_associations_all), number_of_detected_targets_from_previous_frame+number_of_measurements_from_current_frame))

                    for track_index in range(number_of_detected_targets_from_previous_frame):
                        single_target_hypothesis_index_specified_by_previous_step_global_hypothesis=filter_predicted['globHyp'][global_hypothesis_index_from_pevious_frame][track_index] # Readout the single target hypothesis index as specified by the global hypothesis of previous step
                        indices_of_new_hypotheses_generated_from_this_previous_hypothesis = [idx for idx, value in enumerate(filter_updated['tracks'][track_index]['single_target_hypothesis_index_from_previous_frame']) if value == single_target_hypothesis_index_specified_by_previous_step_global_hypothesis]
                        for ith_optimal_option_index, ith_optimal_option_measurement_track_association_vector in enumerate(optimal_associations_all): 
                            
                            indices_of_ith_optimal_option_associated_measurement_list = [idx for idx, value in enumerate(ith_optimal_option_measurement_track_association_vector) if value == track_index] # if this track is part of optimal single target hypothesis          
                            if len(indices_of_ith_optimal_option_associated_measurement_list)==0:
                                if single_target_hypothesis_index_specified_by_previous_step_global_hypothesis == -1:
                                    single_target_hypothesis_index = single_target_hypothesis_index_specified_by_previous_step_global_hypothesis
                                else:
                                    single_target_hypothesis_index = indices_of_new_hypotheses_generated_from_this_previous_hypothesis[0]
                            else:
                                ith_best_optimal_measurement_association_for_this_track = indices_of_ith_optimal_option_associated_measurement_list[0]
                                
                                for new_single_target_hypothesis_index in indices_of_new_hypotheses_generated_from_this_previous_hypothesis[1:]:
                                    
                                    if filter_updated['tracks'][track_index]['measurement_association_from_this_frame'][new_single_target_hypothesis_index]==ith_best_optimal_measurement_association_for_this_track:
                                        
                                        single_target_hypothesis_index=new_single_target_hypothesis_index
                            
                            globHyp_from_current_frame_under_this_globHyp_from_previous_frame[ith_optimal_option_index][track_index]=single_target_hypothesis_index
    
                    for i in range(number_of_measurements_from_current_frame):
                        potential_new_track_index = number_of_detected_targets_from_previous_frame + i
                        for ith_optimal_option_index, ith_optimal_option_vector in enumerate(optimal_associations_all): # get he number of row vectors of opt_indices_trans
                            indices_of_ith_optimal_option_associated_measurement_list = [idx for idx, value in enumerate(ith_optimal_option_vector) if value == potential_new_track_index] # if this track is part of optimal single target hypothesis          
                            if len(indices_of_ith_optimal_option_associated_measurement_list)==0:
                                
                                single_target_hypothesis_index = -1
                            else:
                                single_target_hypothesis_index = 0
                            globHyp_from_current_frame_under_this_globHyp_from_previous_frame[ith_optimal_option_index][potential_new_track_index]=single_target_hypothesis_index 
    
                    for ith_optimal_global_hypothesis in globHyp_from_current_frame_under_this_globHyp_from_previous_frame:
                        globHyp.append(ith_optimal_global_hypothesis)
                filter_updated['globHyp']=globHyp
                if len(weight_of_global_hypothesis_in_log_format)>0:
                    maximum_weight_of_global_hypothesis_in_log_format = np.max(weight_of_global_hypothesis_in_log_format)              
                    globWeight=[np.exp(x-maximum_weight_of_global_hypothesis_in_log_format) for x in weight_of_global_hypothesis_in_log_format]
                    globWeight=globWeight/sum(globWeight)
                else:
                    globWeight = []
                filter_updated['globHypWeight']=globWeight  

        return filter_updated

    def extractStates(self, filter_updated):
        state_extraction_option = self.model['state_extraction_option']
        globHyp=filter_updated['globHyp']
        globHypWeight=filter_updated['globHypWeight']
        number_of_global_hypotheses_at_current_frame = len(globHypWeight)
        number_of_tracks_at_current_frame=len(filter_updated['tracks'])
        
        state_estimate = {}
        mean = []
        rotation = []
        elevation = []
        classification = []
        size = []
        covariance = []
        detection_score = []
        eB_list=[]
        association_history = [[] for i in range(number_of_tracks_at_current_frame)]
        association_counter=[]
        id=[]
        weight=[]

        if number_of_global_hypotheses_at_current_frame>0:
            highest_weight_global_hypothesis_index = np.argmax(globHypWeight)
            highest_weight_global_hypothesis=globHyp[highest_weight_global_hypothesis_index]
            for track_index in range(len(highest_weight_global_hypothesis)):
                single_target_hypothesis_specified_by_global_hypothesis=int(highest_weight_global_hypothesis[track_index])
                if single_target_hypothesis_specified_by_global_hypothesis > -1:

                    eB=filter_updated['tracks'][track_index]['eB'][single_target_hypothesis_specified_by_global_hypothesis]
                    if eB >self.model['eB_estimation_threshold']:
                        mean.append(filter_updated['tracks'][track_index]['meanB'][single_target_hypothesis_specified_by_global_hypothesis])
                        rotation.append(filter_updated['tracks'][track_index]['rotationB'][single_target_hypothesis_specified_by_global_hypothesis])
                        elevation.append(filter_updated['tracks'][track_index]['elevationB'][single_target_hypothesis_specified_by_global_hypothesis])
                        classification.append(filter_updated['tracks'][track_index]['classificationB'][single_target_hypothesis_specified_by_global_hypothesis])
                        eB_list.append(eB)
                        id.append(filter_updated['tracks'][track_index]['idB'][single_target_hypothesis_specified_by_global_hypothesis])
                        covariance.append(filter_updated['tracks'][track_index]['covB'][single_target_hypothesis_specified_by_global_hypothesis])
                        size.append(filter_updated['tracks'][track_index]['sizeB'][single_target_hypothesis_specified_by_global_hypothesis])
                        detection_score.append(filter_updated['tracks'][track_index]['detection_scoreB'][single_target_hypothesis_specified_by_global_hypothesis])
                        associated_measurement=filter_updated['tracks'][track_index]['measurement_association_history'][single_target_hypothesis_specified_by_global_hypothesis]
                        association_history[track_index].append(associated_measurement)
                        weight.append(filter_updated['tracks'][track_index]['giou'][single_target_hypothesis_specified_by_global_hypothesis])
                        association_counter.append(filter_updated['tracks'][track_index]['association_counter'][single_target_hypothesis_specified_by_global_hypothesis])


        state_estimate['mean'] = mean
        state_estimate['covariance'] = covariance
        state_estimate['rotation']=rotation
        state_estimate['elevation']=elevation
        state_estimate['size']=size
        state_estimate['classification']=classification
        state_estimate['id']=id
        state_estimate['detection_score'] = detection_score
        state_estimate['measurement_association_history'] = association_history
        state_estimate['eB']=eB_list
        state_estimate['weight']=weight
        state_estimate['association_counter']=association_counter

        return state_estimate

    def extractStates_with_custom_thr(self, filter_updated, thr):
        state_extraction_option = self.model['state_extraction_option']

        globHyp=filter_updated['globHyp']
        globHypWeight=filter_updated['globHypWeight']
        number_of_global_hypotheses_at_current_frame = len(globHypWeight)
        number_of_tracks_at_current_frame=len(filter_updated['tracks'])
        state_estimate = {}
        mean = []
        rotation = []
        elevation = []
        classification = []
        size = []
        covariance = []
        detection_score = []
        eB_list=[]
        association_all = []
        association_history_all=[[] for x in range(number_of_tracks_at_current_frame)]
        id=[]
        weight=[]

        if number_of_global_hypotheses_at_current_frame>0:
            highest_weight_global_hypothesis_index = np.argmax(globHypWeight)
            highest_weight_global_hypothesis=globHyp[highest_weight_global_hypothesis_index]
            for track_index in range(len(highest_weight_global_hypothesis)):
                single_target_hypothesis_specified_by_global_hypothesis=int(highest_weight_global_hypothesis[track_index])
                if single_target_hypothesis_specified_by_global_hypothesis > -1:
                    eB=filter_updated['tracks'][track_index]['eB'][single_target_hypothesis_specified_by_global_hypothesis]
                    if eB >thr:
                        mean.append(filter_updated['tracks'][track_index]['meanB'][single_target_hypothesis_specified_by_global_hypothesis])
                        rotation.append(filter_updated['tracks'][track_index]['rotationB'][single_target_hypothesis_specified_by_global_hypothesis])
                        elevation.append(filter_updated['tracks'][track_index]['elevationB'][single_target_hypothesis_specified_by_global_hypothesis])
                        classification.append(filter_updated['tracks'][track_index]['classificationB'][single_target_hypothesis_specified_by_global_hypothesis])
                        eB_list.append(eB)
                        id.append(filter_updated['tracks'][track_index]['idB'][single_target_hypothesis_specified_by_global_hypothesis])
                        covariance.append(filter_updated['tracks'][track_index]['covB'][single_target_hypothesis_specified_by_global_hypothesis])
                        size.append(filter_updated['tracks'][track_index]['sizeB'][single_target_hypothesis_specified_by_global_hypothesis])
                        detection_score.append(filter_updated['tracks'][track_index]['detection_scoreB'][single_target_hypothesis_specified_by_global_hypothesis])
                        associated_measurement=filter_updated['tracks'][track_index]['measurement_association_history'][single_target_hypothesis_specified_by_global_hypothesis]
                        association_all.append(associated_measurement)
                        association_history_all[track_index]=filter_updated['tracks'][track_index]['measurement_association_history']
                        
                        weight.append(filter_updated['tracks'][track_index]['giou'][single_target_hypothesis_specified_by_global_hypothesis])


        state_estimate['mean'] = mean
        state_estimate['covariance'] = covariance
        state_estimate['rotation']=rotation
        state_estimate['elevation']=elevation
        state_estimate['size']=size
        state_estimate['classification']=classification
        state_estimate['id']=id
        state_estimate['detection_score'] = detection_score
        state_estimate['measurement_association_history'] = association_history_all
        state_estimate['eB']=eB_list
        state_estimate['weight']=weight

        return state_estimate
    
    def prune(self, filter_updated):
        filter_pruned = copy.deepcopy(filter_updated)
        weightPois=copy.deepcopy(filter_updated['weightPois'])
        global_hypothesis_weights=copy.deepcopy(filter_updated['globHypWeight'])
        globHyp=copy.deepcopy(filter_updated['globHyp'])
        maximum_number_of_global_hypotheses = self.model['maximum_number_of_global_hypotheses']
        eB_threshold = self.model['eB_threshold']
        Poisson_threshold = self.model['T_pruning_Pois']
        MBM_threshold = self.model['T_pruning_MBM']
        indices_to_remove_poisson=[index for index, value in enumerate(weightPois) if value<Poisson_threshold]
        for offset, idx in enumerate(indices_to_remove_poisson):
            del filter_pruned['weightPois'][idx-offset]
            del filter_pruned['rotationPois'][idx-offset]
            del filter_pruned['elevationPois'][idx-offset]
            del filter_pruned['classificationPois'][idx-offset]
            del filter_pruned['sizePois'][idx-offset]
            del filter_pruned['idPois'][idx-offset]
            del filter_pruned['meanPois'][idx-offset]
            del filter_pruned['covPois'][idx-offset]
            del filter_pruned['detection_scorePois'][idx-offset]
        indices_to_keep_global_hypotheses=[index for index, value in enumerate(global_hypothesis_weights) if value>MBM_threshold]
        weights_after_pruning_before_capping=[global_hypothesis_weights[x] for x in indices_to_keep_global_hypotheses]
        globHyp_after_pruning_before_capping=[globHyp[x] for x in indices_to_keep_global_hypotheses]

        weight_after_pruning_negative_value = [-x for x in weights_after_pruning_before_capping]
        index_of_ranked_global_hypothesis_weights_in_descending_order=np.argsort(weight_after_pruning_negative_value)
        if len(weights_after_pruning_before_capping)>maximum_number_of_global_hypotheses:
            indices_to_keep_global_hypotheses_capped = index_of_ranked_global_hypothesis_weights_in_descending_order[:maximum_number_of_global_hypotheses]
        else:
            indices_to_keep_global_hypotheses_capped=index_of_ranked_global_hypothesis_weights_in_descending_order[:len((weights_after_pruning_before_capping))]
        
        
        globHyp_after_pruning = [copy.deepcopy(globHyp_after_pruning_before_capping[x]) for x in indices_to_keep_global_hypotheses_capped] 
        weights_after_pruning = [copy.deepcopy(weights_after_pruning_before_capping[x]) for x in indices_to_keep_global_hypotheses_capped]
        weights_after_pruning=[x/np.sum(weights_after_pruning) for x in weights_after_pruning]
        globHyp_after_pruning=np.array(globHyp_after_pruning)
        weights_after_pruning=np.array(weights_after_pruning)

        if len(globHyp_after_pruning)>0:            
            for track_index in range(len(filter_pruned['tracks'])):     
                indices_of_single_target_hypotheses_to_be_marked=[index for index,value in enumerate(filter_pruned['tracks'][track_index]['eB']) if value < eB_threshold]
                for single_target_hypothesis_to_be_marked_idx in indices_of_single_target_hypotheses_to_be_marked:
                    for index_of_single_target_hypothesis_in_global_hypothesis,single_target_hypothesis_in_global_hypothesis in enumerate(globHyp_after_pruning[:,track_index]):
                        if single_target_hypothesis_in_global_hypothesis==single_target_hypothesis_to_be_marked_idx:
                            globHyp_after_pruning[:,track_index][index_of_single_target_hypothesis_in_global_hypothesis]=-1
            if len(globHyp_after_pruning) > 0:
                tracks_to_be_removed = [x for x in range(len(globHyp_after_pruning[0])) if np.sum(globHyp_after_pruning[:,x]) == -len(globHyp_after_pruning)]
            else:
                tracks_to_be_removed=[]
            if len(tracks_to_be_removed)>0:
                for offset, track_index_to_be_removed in enumerate(tracks_to_be_removed):
                    del filter_pruned['tracks'][track_index_to_be_removed-offset]
                globHyp_after_pruning = np.delete(globHyp_after_pruning, tracks_to_be_removed, axis=1)
            for track_index in range(len(filter_pruned['tracks'])):
                single_target_hypothesis_indices_to_be_removed = []            
                number_of_single_target_hypothesis =len(filter_pruned['tracks'][track_index]['eB'])
                valid_single_target_hypothesis_for_this_track = globHyp_after_pruning[:,track_index]

                for single_target_hypothesis_index in range(number_of_single_target_hypothesis):

                    if single_target_hypothesis_index not in valid_single_target_hypothesis_for_this_track:

                        single_target_hypothesis_indices_to_be_removed.append(single_target_hypothesis_index)
                if len(single_target_hypothesis_indices_to_be_removed)>0:
                    for offset, index in enumerate(single_target_hypothesis_indices_to_be_removed):
                        del filter_pruned['tracks'][track_index]['eB'][index-offset]
                        del filter_pruned['tracks'][track_index]['meanB'][index-offset]
                        del filter_pruned['tracks'][track_index]['covB'][index-offset]
                        del filter_pruned['tracks'][track_index]['rotationB'][index-offset]
                        del filter_pruned['tracks'][track_index]['elevationB'][index-offset]
                        del filter_pruned['tracks'][track_index]['classificationB'][index-offset]
                        del filter_pruned['tracks'][track_index]['sizeB'][index-offset]
                        del filter_pruned['tracks'][track_index]['idB'][index-offset]
                        del filter_pruned['tracks'][track_index]['detection_scoreB'][index-offset]
                        del filter_pruned['tracks'][track_index]['weight_of_single_target_hypothesis_in_log_format'][index-offset]
                        del filter_pruned['tracks'][track_index]['giou'][index-offset]
                        del filter_pruned['tracks'][track_index]['single_target_hypothesis_index_from_previous_frame'][index-offset]
                        del filter_pruned['tracks'][track_index]['measurement_association_history'][index-offset]
                        del filter_pruned['tracks'][track_index]['measurement_association_from_this_frame'][index-offset]
                        del filter_pruned['tracks'][track_index]['association_counter'][index-offset]
                if len(single_target_hypothesis_indices_to_be_removed)>0:
                    for global_hypothesis_index, global_hypothesis_vector in enumerate(globHyp_after_pruning):
                        single_target_hypothesis_specified_by_the_global_hypothesis = global_hypothesis_vector[track_index]
                        single_target_hypotheses_removed_before_this_single_taget_hypothesis = [x for x in single_target_hypothesis_indices_to_be_removed if x<single_target_hypothesis_specified_by_the_global_hypothesis]
                        subtraction=len(single_target_hypotheses_removed_before_this_single_taget_hypothesis)
                        globHyp_after_pruning[global_hypothesis_index][track_index]-=subtraction
      
            globHyp_unique, indices= np.unique(globHyp_after_pruning, axis=0, return_index = True)

            duplicated_indices = [x for x in range(len(globHyp_after_pruning)) if x not in indices]

            if len(globHyp_unique)!=len(globHyp_after_pruning):
                weights_unique=np.zeros(len(globHyp_unique))
                for i in range(len(globHyp_unique)):

                    weights_unique[i] = global_hypothesis_weights[indices[i]]
                    for j in duplicated_indices:
                        if list(globHyp_after_pruning[j]) == list(globHyp_unique[i]):

                            weights_unique[i]+=global_hypothesis_weights[j]
    
                globHyp_after_pruning=globHyp_unique
                weights_after_pruning=weights_unique
                weights_after_pruning/sum(weights_after_pruning)
        
        filter_pruned['globHyp']=globHyp_after_pruning
        filter_pruned['globHypWeight']=weights_after_pruning
        return filter_pruned