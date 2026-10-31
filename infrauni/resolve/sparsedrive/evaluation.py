"""Map SparseDrive query outputs to the single InfraUni RESOLVE evaluator."""
import numpy as np
from infrauni.eval.resolve_detection import CLASS_NAMES,evaluate_boxes
from infrauni.eval.resolve_forecast import evaluate_end_to_end

def ground_truth(infos):
    out={}
    for d in infos:
        out[d['token']]=[dict(translation=b[:3].tolist(),size=b[3:6].tolist(),yaw=float(b[6]),velocity=b[7:9].tolist(),name=CLASS_NAMES[int(d['gt_labels'][i])],future_xy=d['gt_future'][i],future_mask=d['gt_future_mask'][i]) for i,b in enumerate(d['gt_boxes'])]
    return out

def normalize(result):
    if 'img_bbox' in result:result=result['img_bbox']
    boxes=result['boxes_3d'].numpy();scores=result['scores_3d'].numpy();labels=result['labels_3d'].numpy()
    trajectories=result.get('trajs_3d');modes=result.get('trajs_score');items=[]
    for i,b in enumerate(boxes):
        item=dict(translation=b[:3].tolist(),size=b[3:6].tolist(),yaw=float(b[6]),velocity=b[7:9].tolist(),name=CLASS_NAMES[int(labels[i])],score=float(scores[i]))
        if trajectories is not None:item.update(future_xy=trajectories[i].numpy(),mode_probs=modes[i].numpy())
        items.append(item)
    return items

def evaluate(infos,predictions,motion=True):
    truth=ground_truth(infos)
    assert set(truth)==set(predictions),'Every validation frame must appear, including frames with no predictions'
    result={'detection':evaluate_boxes(truth,predictions),'frames':len(infos)}
    if motion:result['forecast']=evaluate_end_to_end(truth,predictions)
    result['protocol_details']={'source':'InfraUni shared evaluator','camera_only':True,'prediction_steps':10,'prediction_hz':2,'prediction_modes':6,'motion':motion}
    return result
