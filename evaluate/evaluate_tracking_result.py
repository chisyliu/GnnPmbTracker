import os, numpy as np, argparse, json, multiprocessing
import json
import os
import time
from typing import Tuple, List, Dict, Any
import numpy as np
from nuscenes import NuScenes
from evaluate.util.utils import TrackingEvaluation,recall_metric_curve, summary_plot,AVG_METRIC_MAP, MOT_METRIC_MAP, LEGACY_METRICS, TrackingMetrics, TrackingMetricDataList, TrackingConfig, TrackingBox, \
    TrackingMetricData, create_tracks,print_final_metrics, render_for_the_best_threshold,config_factory,load_prediction, load_gt, add_center_dist, filter_eval_boxes

class TrackingEval:
    def __init__(self,
                 config: TrackingConfig,
                 result_path: str,
                 eval_set: str,
                 output_dir: str,
                 nusc_version: str,
                 nusc_dataroot: str,
                 verbose: bool = True,
                 render_classes: List[str] = None):
        self.cfg = config
        self.result_path = result_path
        self.eval_set = eval_set
        self.output_dir = output_dir
        self.verbose = verbose
        self.render_classes = render_classes

        assert os.path.exists(result_path), 'Error: The result file does not exist!'

        # Make dirs.
        self.plot_dir = os.path.join(self.output_dir, 'plots')
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.isdir(self.plot_dir):
            os.makedirs(self.plot_dir)

        nusc = NuScenes(version=nusc_version, verbose=verbose, dataroot=nusc_dataroot)
        self.nusc = nusc

        if verbose:
            print('Initializing nuScenes tracking evaluation')
        pred_boxes, self.meta = load_prediction(self.result_path, self.cfg.max_boxes_per_sample, TrackingBox,
                                                verbose=verbose)
        gt_boxes = load_gt(nusc, self.eval_set, TrackingBox, verbose=verbose)

        assert set(pred_boxes.sample_tokens) == set(gt_boxes.sample_tokens), \
            "Samples in split don't match samples in predicted tracks."

        pred_boxes = add_center_dist(nusc, pred_boxes)
        gt_boxes = add_center_dist(nusc, gt_boxes)

        if verbose:
            print('Filtering tracks')
        pred_boxes = filter_eval_boxes(nusc, pred_boxes, self.cfg.class_range, verbose=verbose)
        if verbose:
            print('Filtering ground truth tracks')
        gt_boxes = filter_eval_boxes(nusc, gt_boxes, self.cfg.class_range, verbose=verbose)

        self.sample_tokens = gt_boxes.sample_tokens

        self.tracks_gt = create_tracks(gt_boxes, nusc, self.eval_set, gt=True)
        self.tracks_pred = create_tracks(pred_boxes, nusc, self.eval_set, gt=False)

    def evaluate(self) -> Tuple[TrackingMetrics, TrackingMetricDataList]:
        """
        Performs the actual evaluation.
        :return: A tuple of high-level and the raw metric data.
        """
        start_time = time.time()
        metrics = TrackingMetrics(self.cfg)

        if self.verbose:
            print('Accumulating metric data...')
        metric_data_list = TrackingMetricDataList()

        def accumulate_class(curr_class_name):
            curr_ev = TrackingEvaluation(self.nusc,self.tracks_gt, self.tracks_pred, curr_class_name, self.cfg.dist_fcn_callable,
                                         self.cfg.dist_th_tp, self.cfg.min_recall,
                                         num_thresholds=TrackingMetricData.nelem,
                                         metric_worst=self.cfg.metric_worst,
                                         verbose=self.verbose,
                                         output_dir=self.output_dir,
                                         render_classes=self.render_classes)
            curr_md = curr_ev.accumulate()
            metric_data_list.set(curr_class_name, curr_md)

        for class_name in self.cfg.class_names:
            accumulate_class(class_name)

        if self.verbose:
            print('Calculating metrics...')

        for class_name in self.cfg.class_names:

            md = metric_data_list[class_name]
            if np.all(np.isnan(md.mota)):
                best_thresh_idx = None
            else:
                best_thresh_idx = np.nanargmax(md.mota)
            
            if best_thresh_idx!=None:
                best_threshold=md.confidence[best_thresh_idx]
                render_for_the_best_threshold(self.nusc, class_name, best_threshold, self.tracks_gt, self.tracks_pred,self.render_classes,self.output_dir,self.cfg.dist_th_tp)

            if best_thresh_idx is not None:
                for metric_name in MOT_METRIC_MAP.values():
                    if metric_name == '':
                        continue
                    value = md.get_metric(metric_name)[best_thresh_idx]
                    metrics.add_label_metric(metric_name, class_name, value)

            for metric_name in AVG_METRIC_MAP.keys():
                values = np.array(md.get_metric(AVG_METRIC_MAP[metric_name]))
                assert len(values) == TrackingMetricData.nelem

                if np.all(np.isnan(values)):
                    value = np.nan
                else:
                    np.all(values[np.logical_not(np.isnan(values))] >= 0)
                    values[np.isnan(values)] = self.cfg.metric_worst[metric_name]
                    value = float(np.nanmean(values))
                metrics.add_label_metric(metric_name, class_name, value)

        metrics.add_runtime(time.time() - start_time)

        return metrics, metric_data_list

    def render(self, md_list: TrackingMetricDataList) -> None:
        if self.verbose:
            print('Rendering curves')

        def savepath(name):
            return os.path.join(self.plot_dir, name + '.pdf')

        summary_plot(self.cfg, md_list, savepath=savepath('summary'))

        for metric_name in LEGACY_METRICS:
            recall_metric_curve(self.cfg, md_list, metric_name, savepath=savepath('%s' % metric_name))

    def visualization_and_evaluation_of_tracking_results(self, render_curves: bool = True) -> Dict[str, Any]:
        metrics, metric_data_list = self.evaluate()

        if self.verbose:
            print('Saving metrics to: %s' % self.output_dir)
        metrics_summary = metrics.serialize()
        metrics_summary['meta'] = self.meta.copy()
        with open(os.path.join(self.output_dir, 'metrics_summary.json'), 'w') as f:
            json.dump(metrics_summary, f, indent=2)
        with open(os.path.join(self.output_dir, 'metrics_details.json'), 'w') as f:
            json.dump(metric_data_list.serialize(), f, indent=2)

        if self.verbose:
            print_final_metrics(metrics)

        if render_curves:
            self.render(metric_data_list)

        return metrics_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate nuScenes tracking results.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--result_path', default='/home/Desktop/evaluate/val_submission.json',type=str, help='The submission as a JSON file.')
    parser.add_argument('--output_dir', type=str, default='/home/Desktop/evaluate/nuscenes-metrics_trainval',
                        help='Folder to store result metrics, graphs and example visualizations.')
    parser.add_argument('--eval_set', type=str, default='val',
                        help='Which dataset split to evaluate on, train, val or test or mini_val.')
    parser.add_argument('--dataroot', type=str, default='/home/Desktop/mmdetection3d/data/nuscenes',
                        help='Default nuScenes data directory.')
    parser.add_argument('--version', type=str, default='v1.0-trainval',
                        help='Which version of the nuScenes dataset to evaluate on, e.g. v1.0-trainval.')
    parser.add_argument('--config_path', type=str,default='',
                        help='Path to the configuration file.'
                             'If no path given, the NIPS 2019 configuration will be used.')
    parser.add_argument('--render_curves', type=int, default=1,
                        help='Whether to render statistic curves to disk.')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Whether to print to stdout.')
    parser.add_argument('--render_classes', type=str, default='', nargs='+',
                        help='For which classes we render tracking results to disk.')
    args = parser.parse_args()

    result_path_ = os.path.expanduser(args.result_path)
    output_dir_ = os.path.expanduser(args.output_dir)
    eval_set_ = args.eval_set
    dataroot_ = args.dataroot
    version_ = args.version
    config_path = args.config_path
    render_curves_ = bool(args.render_curves)
    verbose_ = bool(args.verbose)
    render_classes_ = args.render_classes

    if config_path == '':
        cfg_ = config_factory('/home/Desktop/Radar_Perception_Project/Project_5/configs/tracking_config.json')
    else:
        with open(config_path, 'r') as _f:
            cfg_ = TrackingConfig.deserialize(json.load(_f))

    nusc_eval = TrackingEval(config=cfg_, result_path=result_path_, eval_set=eval_set_, output_dir=output_dir_,
                             nusc_version=version_, nusc_dataroot=dataroot_, verbose=verbose_,
                             render_classes=render_classes_)
    nusc_eval.visualization_and_evaluation_of_tracking_results(render_curves=render_curves_)

