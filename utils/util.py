import numpy as np

def read_class_name(class_file_name):
    names = {}
    with open(class_file_name, 'r') as f:
        for id, name in enumerate(f):
            names[id] = name.strip('\n')
    return names


def object_filtering(bbox_xywh, original_w, original_h, h_factor = 0.03, w_factor = 0.01):
    """
        bbox_xywh = top left x,y, width, height
    """
    
    mask = np.logical_and(bbox_xywh[:, 3] > original_h * h_factor,
                        bbox_xywh[:, 2] > original_w * w_factor)
    accepted = np.where(mask)
    
    rejected = np.where(np.logical_not(mask))
    
    return accepted, rejected 


def region_filtering_by_relative_position(bbox_xywh, original_h, upper_limit = 0., lower_limit = 0.):
    """
        bbox_xywh = top left x,y, width, height
    """
    center_xy = (bbox_xywh[:, :2]+bbox_xywh[:, 2:]/2)
    
    mask = np.logical_and(center_xy[:, 1] > original_h * upper_limit,
                         center_xy[:, 1] < original_h * (1.-lower_limit))
    
    accepted = np.where(mask)
    
    rejected = np.where(np.logical_not(mask))
    
    return accepted, rejected

def frame2time(frame, fps):
    
    second = float(frame) / float(fps)
    
    remainder = second - float(int(second))
    
    second = int(second)
    
    hour = second // 3600
    
    second %= 3600
    
    minute = second // 60
    
    second %= 60
    
    return f"{hour}h{minute}m{second}.{str(remainder)[2:3]}s"
    