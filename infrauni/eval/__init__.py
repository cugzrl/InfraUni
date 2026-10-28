"""Evaluation APIs. Legacy nuScenes dependencies load only on use."""
def evaluate_detection(*args, **kwargs):
    from .nusc_det import evaluate_detection as fn
    return fn(*args, **kwargs)

def summarize_forecast(*args, **kwargs):
    from .protocol import summarize_forecast as fn
    return fn(*args, **kwargs)
