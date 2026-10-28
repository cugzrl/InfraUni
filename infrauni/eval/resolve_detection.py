"""RESOLVE official AP/TP curves; paper NDS9 is the primary score.
Source: third_party/RESOLVE/pcdet/datasets/carla/sunlakes_utils.py.
No nuScenes radial filtering or cone/barrier TP exclusions are applied.
"""
import time
import numpy as np
from nuscenes.eval.detection.data_classes import DetectionMetrics, DetectionBox, DetectionMetricDataList, DetectionMetricData
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.common.utils import center_distance, scale_iou, yaw_diff, velocity_l2, attr_acc, cummean
from nuscenes.eval.detection.config import config_factory

CLASS_NAMES = ('car','truck','construction_vehicle','bus','trailer','barrier','motorcycle','bicycle','pedestrian','traffic_cone')
def accumulate(gt_boxes,
               pred_boxes,
               class_name,
               dist_fcn,
               dist_th):
    # Count the positives.
    npos = len([1 for gt_box in gt_boxes.all if gt_box.detection_name == class_name])
    # For missing classes in the GT, return a data structure corresponding to no predictions.
    if npos == 0:
        return DetectionMetricData.no_predictions()
    # Organize the predictions in a single list.
    pred_boxes_list = [box for box in pred_boxes.all if box.detection_name == class_name]
    pred_confs = [box.detection_score for box in pred_boxes_list]
    # Sort by confidence.
    sortind = [i for (v, i) in sorted((v, i) for (i, v) in enumerate(pred_confs))][::-1]
    # Do the actual matching.
    tp = []  # Accumulator of true positives.
    fp = []  # Accumulator of false positives.
    conf = []  # Accumulator of confidences.
    # match_data holds the extra metrics we calculate for each match.
    match_data = {'trans_err': [],
                  'vel_err': [],
                  'scale_err': [],
                  'orient_err': [],
                  'attr_err': [],
                  'conf': []}
    # ---------------------------------------------
    # Match and accumulate match data.
    # ---------------------------------------------
    taken = set()  # Initially no gt bounding box is matched.
    for ind in sortind:
        pred_box = pred_boxes_list[ind]
        min_dist = np.inf
        match_gt_idx = None
        for gt_idx, gt_box in enumerate(gt_boxes[pred_box.sample_token]):

            # Find closest match among ground truth boxes
            if gt_box.detection_name == class_name and not (pred_box.sample_token, gt_idx) in taken:
                this_distance = dist_fcn(gt_box, pred_box)
                if this_distance < min_dist:
                    min_dist = this_distance
                    match_gt_idx = gt_idx
        # If the closest match is close enough according to threshold we have a match!
        is_match = min_dist < dist_th
        if is_match:
            taken.add((pred_box.sample_token, match_gt_idx))
            #  Update tp, fp and confs.
            tp.append(1)
            fp.append(0)
            conf.append(pred_box.detection_score)
            # Since it is a match, update match data also.
            gt_box_match = gt_boxes[pred_box.sample_token][match_gt_idx]
            match_data['trans_err'].append(center_distance(gt_box_match, pred_box))
            match_data['vel_err'].append(velocity_l2(gt_box_match, pred_box))
            match_data['scale_err'].append(1 - scale_iou(gt_box_match, pred_box))
            # Barrier orientation is only determined up to 180 degree. (For cones orientation is discarded later)
            period = np.pi if class_name == 'barrier' else 2 * np.pi
            match_data['orient_err'].append(yaw_diff(gt_box_match, pred_box, period=period))
            match_data['attr_err'].append(1 - attr_acc(gt_box_match, pred_box))
            match_data['conf'].append(pred_box.detection_score)
        else:
            # No match. Mark this as a false positive.
            tp.append(0)
            fp.append(1)
            conf.append(pred_box.detection_score)
    # Check if we have any matches. If not, just return a "no predictions" array.
    if len(match_data['trans_err']) == 0:
        return DetectionMetricData.no_predictions()
    # ---------------------------------------------
    # Calculate and interpolate precision and recall
    # --------------------------------------------
    # Accumulate.
    tp = np.cumsum(tp).astype(float)
    fp = np.cumsum(fp).astype(float)
    conf = np.array(conf)
    # Calculate precision and recall.
    prec = tp / (fp + tp)
    rec = tp / float(npos)
    rec_interp = np.linspace(0, 1, DetectionMetricData.nelem)  # 101 steps, from 0% to 100% recall.
    prec = np.interp(rec_interp, rec, prec, right=0)
    conf = np.interp(rec_interp, rec, conf, right=0)
    rec = rec_interp
    # ---------------------------------------------
    # Re-sample the match-data to match, prec, recall and conf.
    # ---------------------------------------------
    for key in match_data.keys():
        if key == "conf":
            continue  # Confidence is used as reference to align with fp and tp. So skip in this step.
        else:
            # For each match_data, we first calculate the accumulated mean.
            tmp = cummean(np.array(match_data[key]))
            # Then interpolate based on the confidences. (Note reversing since np.interp needs increasing arrays)
            match_data[key] = np.interp(conf[::-1], match_data['conf'][::-1], tmp[::-1])[::-1]
    # ---------------------------------------------
    # Done. Instantiate MetricData and return
    # ---------------------------------------------
    return DetectionMetricData(recall=rec,
                               precision=prec,
                               confidence=conf,
                               trans_err=match_data['trans_err'],
                               vel_err=match_data['vel_err'],
                               scale_err=match_data['scale_err'],
                               orient_err=match_data['orient_err'],
                               attr_err=match_data['attr_err'])


def cal_metrics(cfg, gt_boxes, pred_boxes):
    start_time = time.time()
    metric_data_list = DetectionMetricDataList()
    for class_name in cfg.class_names:
        for dist_th in cfg.dist_ths:
            md = accumulate(gt_boxes, pred_boxes, class_name, cfg.dist_fcn_callable, dist_th)
            metric_data_list.set(class_name, dist_th, md)
    # -----------------------------------
    # Step 2: Calculate metrics from the data.
    # -----------------------------------
    TP_METRICS = ['trans_err', 'scale_err', 'orient_err', 'vel_err', 'attr_err']
    metrics = DetectionMetrics(cfg)
    for class_name in cfg.class_names:
        # Compute APs.
        for dist_th in cfg.dist_ths:
            metric_data = metric_data_list[(class_name, dist_th)]
            ap = calc_ap(metric_data, cfg.min_recall, cfg.min_precision)
            metrics.add_label_ap(class_name, dist_th, ap)
        # Compute TP metrics.
        for metric_name in TP_METRICS:
            metric_data = metric_data_list[(class_name, cfg.dist_th_tp)]
            #if class_name in ['traffic_cone'] and metric_name in ['attr_err', 'vel_err', 'orient_err']:
            #    tp = np.nan
            #elif class_name in ['barrier'] and metric_name in ['attr_err', 'vel_err']:
            #    tp = np.nan
            #else:
            tp = calc_tp(metric_data, cfg.min_recall, metric_name)
            metrics.add_label_tp(class_name, metric_name, tp)
    # Compute evaluation time.
    metrics.add_runtime(time.time() - start_time)
    return metrics, metric_data_list


def calc_ap(md: DetectionMetricData, min_recall: float, min_precision: float) -> float:
    """ Calculated average precision. """
    assert 0 <= min_precision < 1
    assert 0 <= min_recall <= 1
    prec = np.copy(md.precision)
    prec = prec[round(100 * min_recall) + 1:]  # Clip low recalls. +1 to exclude the min recall bin.
    prec -= min_precision  # Clip low precision
    prec[prec < 0] = 0
    return float(np.mean(prec)) / (1.0 - min_precision)


def calc_tp(md: DetectionMetricData, min_recall: float, metric_name: str) -> float:
    """ Calculates true positive errors. """
    first_ind = round(100 * min_recall) + 1  # +1 to exclude the error at min recall.
    last_ind = md.max_recall_ind  # First instance of confidence = 0 is index of max achieved recall.
    if last_ind < first_ind:
        return 1.0  # Assign 1 here. If this happens for all classes, the score for that TP metric will be 0.
    else:
        return float(np.mean(getattr(md, metric_name)[first_ind: last_ind + 1]))  # +1 to include error at max recall.



def evaluate_boxes(gt_by_token, pred_by_token):
    unknown=set(pred_by_token)-set(gt_by_token)
    if unknown:
        raise ValueError('Predictions contain unknown frame tokens: '+str(sorted(unknown)[:3]))
    def convert(group, prediction):
        out=EvalBoxes()
        for token in gt_by_token:
            boxes=[]
            for b in group.get(token, []):
                if len(group.get(token, [])) > 500 and prediction:
                    raise ValueError('RESOLVE permits at most 500 predictions per frame')
                yaw=float(b['yaw'])
                boxes.append(DetectionBox(sample_token=token, translation=b['translation'],size=b['size'],
                    rotation=[float(np.cos(yaw/2)),0,0,float(np.sin(yaw/2))],velocity=b.get('velocity',[0.,0.]),
                    detection_name=b['name'],detection_score=float(b['score']) if prediction else -1.,attribute_name=''))
            out.add_boxes(token,boxes)
        return out
    metrics,_=cal_metrics(config_factory('detection_cvpr_2019'),convert(gt_by_token,False),convert(pred_by_token,True))
    result=metrics.serialize()
    result['protocol_details']={'classes':list(CLASS_NAMES),'distance_thresholds_m':[0.5,1,2,4],'tp_threshold_m':2,'radial_class_filter':False,'exclude_cone_barrier_tp':False,'nds_denominator':9,'attribute_term_in_primary_nds':False}
    result['NDS_legacy10']=result['nd_score']
    result['nd_score']=(5*result['mean_ap']+sum(max(0.,1-result['tp_errors'][k]) for k in ('trans_err','scale_err','orient_err','vel_err')))/9
    result.update(mAP=result['mean_ap'],NDS=result['nd_score'],protocol='resolve_official_curves_paper_nds9_v1')
    return result

evaluate_boxes_flat=evaluate_boxes
