import time
from absl.flags import FLAGS
from absl import app, flags
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchvision
import torch
from collections import defaultdict

from utils import util

# reid
from reid import generate_detection_fast_reid as gdet

# yolo v8
from ultralytics import YOLO

# deep sort
from deep_sort import nn_matching, preprocessing
from deep_sort.tracker import Tracker
from deep_sort.detection import Detection

flags.DEFINE_string("reid_weights", "./weights/reid/veriwild_dynamic.onnx", "ReID pretrained model")
flags.DEFINE_string("weights", './weights/yolov8m.pt', "path to weights file")
flags.DEFINE_string("video", "./data/section_cam.mp4", "path to input video or set to 0 for webcam")
flags.DEFINE_string("target", "./data/number_recog000109.jpg", "target image")
flags.DEFINE_string("output", "./output/find_result.mp4", "path to output video")
flags.DEFINE_string("output_format", "mp4v", "codec used in VideoWriter when savinng video to file")
flags.DEFINE_integer('size', 640, "resize images to")
flags.DEFINE_float("iou", 0.5, "iou threshold")
flags.DEFINE_float("conf", 0.5, "score threshold")
flags.DEFINE_boolean("dont_show", False, "dont_show video output")
flags.DEFINE_string("video_mask", "masking_video1.jpg", "path to input video mask")
flags.DEFINE_float("max_cosine_distance", 0.4, "cosine distance threshold")
flags.DEFINE_float("frame_skip", 0.9, "frame skip rate")
flags.DEFINE_float("object_ignore", 0.03, "ignore a object if the object size smaller than frame_height * value")
flags.DEFINE_float("upper_limit", 0., "ignore a object if center of the object in upper limit")
flags.DEFINE_float("lower_limit", 0., "ignore a object if center of the object in lower limit")
flags.DEFINE_boolean("plot_tracking", False, "True if don't want tracking non-matched car")
flags.DEFINE_float("min_cosine_distance", 0.6, "unmatch if distance greater than threshold")
flags.DEFINE_boolean("debug", False, "for debug")
def main(_argv):
    
    # parameters
    max_cosine_distance = FLAGS.max_cosine_distance
    nms_max_overlap = 1.0
    nn_budget = None
    
    # initialize deep sort
    model_filename = FLAGS.reid_weights
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    
    # initialize tracker
    tracker = Tracker(metric, 0.8)
    
    video_path = FLAGS.video
    input_size = FLAGS.size
    
    # load yolo model
    yolo = YOLO(FLAGS.weights)
    
    
    # read all class name
    class_names = util.read_class_name(f"./utils/coco.names")
    allowed_classes = ['car', 'truck']       
    
    # process target image file
    # Only Use Biggest Object in Image
    
    target_img = cv2.imread(FLAGS.target)
    target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
    
    candidates = yolo(target_img, iou = FLAGS.iou, conf = FLAGS.conf, verbose=False)[0]
    max_candidate = -1
    max_size = 0
    for i, candidate in enumerate(candidates):
        _, _, w, h = candidate.boxes.xywh.cpu().numpy().squeeze()
        class_name = class_names[int(candidate.boxes.cls)]
        
        if class_name in allowed_classes and w*h > max_size:
            max_size = w*h
            max_candidate = i
    
    boxes = candidates.boxes.xyxy.cpu().numpy()        
    boxes[:, 2:] = candidates.boxes.xywh[:, 2:].cpu().numpy() 
    target_feature = encoder(target_img, boxes[[max_candidate]])
    
    min_similarity = 1.0
    
    min_track_id = -1
    
    appear_in_frame = defaultdict(list)
    
    
    # begin video capture
    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)
    
    # get video mask if needed
    if FLAGS.video_mask:
        mask = cv2.imread(FLAGS.video_mask)//255
    
    out = None
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # by default VideoCapture returns float instead of int
    fps = int(round(vid.get(cv2.CAP_PROP_FPS)*(1.0 - FLAGS.frame_skip)))
    # get video ready to save locally if flag is set
    if FLAGS.output:
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))
    
    last_frame_num = vid.get(cv2.CAP_PROP_FRAME_COUNT)
    
    total_frame = int(last_frame_num *(1.0 - FLAGS.frame_skip))
    
    per_frame = [int(last_frame_num / total_frame), int(round(last_frame_num / total_frame))]
    
    max_instance = 0
    # while video is running
    for frame_num in tqdm(range(total_frame), desc="Frame #"):
        
        success, frame = vid.read()       
        
        for _ in range(per_frame[frame_num%2]-1):
            success, _ = vid.read()
            if not success:
                print("Video has ended or failed")
                break
            
        
        if success:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            print("Video has ended or failed")
            break
        
        if FLAGS.video_mask:
            image_data = cv2.resize(frame * mask, (input_size, input_size))
        else:
            image_data = cv2.resize(frame, (input_size, input_size))
        
        # image_data /= 255.
        # image_data = image_data[np.newaxis, ...].astype(np.float32)
        
        # yolo
        results = yolo(image_data, iou = FLAGS.iou, conf = FLAGS.conf, verbose=False)[0]
        
        indices = torchvision.ops.nms(results.boxes.xyxy, results.boxes.conf, FLAGS.iou).cpu().numpy()
        
        factor = np.array([width, height, width, height])[np.newaxis, ...]
       
        boxes = results.boxes.xyxyn.cpu().numpy()        
        boxes[:, 2:] = results.boxes.xywhn[:, 2:].cpu().numpy() 

        boxes = (boxes*factor).astype(np.uint16)[indices]
        scores = results.boxes.conf[indices]
        classes = results.boxes.cls[indices]
        accepted, rejected = util.region_filtering_by_relative_position(boxes, height,
                                                                        upper_limit=FLAGS.upper_limit,
                                                                        lower_limit=FLAGS.lower_limit)

        rBoxes_pos = boxes[rejected]
        rScores_pos = scores[rejected]
        rClasses_pos = classes[rejected]
        
        boxes = boxes[accepted]
        scores = scores[accepted]
        classes = classes[accepted]
        
        accepted, rejected = util.object_filtering(boxes, width, height, FLAGS.object_ignore)

        rBoxes_obj = boxes[rejected]
        rScores_obj = scores[rejected]
        rClasses_obj = classes[rejected]

        boxes = boxes[accepted]
        scores = scores[accepted]
        classes = classes[accepted]
        
        
        rejected_boxes = np.concatenate([rBoxes_pos, rBoxes_obj], axis=0)
        rejected_scores = torch.cat([rScores_pos, rScores_obj], axis=0)
        rejected_classes = torch.cat([rClasses_pos, rClasses_obj], axis=0)
        

        names = []
        indx = []
        for i in range(len(classes)):
            class_indx = int(classes[i])
            class_name = class_names[class_indx]
            if class_name in allowed_classes:
                indx.append(i)
                names.append('car')
        bboxes = boxes[indx]
        scores = scores[indx]

        rejected_names = []
        rejected_indx = []
        for i, class_idx in enumerate(rejected_classes):
            class_indx = int(class_idx)
            class_name = class_names[class_indx]
            if class_name in allowed_classes:
                rejected_indx.append(i)
                rejected_names.append('car')
        rejected_bboxes = rejected_boxes[rejected_indx]
        rejected_scores = rejected_scores[rejected_indx].cpu().numpy()


        # encode yolo detections and feed to tracker
        
        features = encoder(frame, bboxes)
        detections = [Detection(bbox, score, class_name, feature)
                      for bbox, score, class_name, feature
                      in zip(bboxes, scores, names, features)]
        
        # initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]
        
        # # run nms
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices] 
        
        # Call the tracker
        tracker.predict()
        tracker.update(detections)
        
        # Gallery
        
        gallery_idxs = []
        gallery_feat = None
        gallery_bbox = []
        gallery_track_ids = []
        for i, gallery in enumerate(tracker.tracks):
            if not gallery.is_confirmed() or gallery.time_since_update > 1:
                continue
            gallery_idxs.append(i)
            gallery_bbox.append(gallery.to_tlwh())
            gallery_track_ids.append(gallery.track_id)
        
        if gallery_bbox:
            
            gallery_bbox = np.stack(gallery_bbox)
            gallery_feat = encoder(frame, gallery_bbox)
            
            cosine_distance = nn_matching._cosine_distance(target_feature, gallery_feat)
            
            argmin = np.argmin(cosine_distance)
            gallery_idx = gallery_idxs[argmin]
            gallery_track_idx = gallery_track_ids[argmin]
            
            if cosine_distance[:, argmin] < FLAGS.min_cosine_distance:
                
                tracker.tracks[gallery_idx].matched = True
                
                # print()
                # print(cosine_distance[:, argmin])
                
                if min_similarity > cosine_distance[:, argmin]:
                    
                    min_similarity = cosine_distance[:, argmin]
                    min_track_id = gallery_track_idx
                
                if len(appear_in_frame[gallery_track_idx]) == 0:
                    appear_in_frame[gallery_track_idx].append(util.frame2time(frame_num, fps))
                    appear_in_frame[gallery_track_idx].append(util.frame2time(frame_num, fps))
                else:
                    appear_in_frame[gallery_track_idx][1] = util.frame2time(frame_num, fps)       
        
        if FLAGS.plot_tracking:
            
            # plot rejected objects
            for i, rejected_bbox in enumerate(rejected_bboxes):
                
                color = (255, 255, 255)
                
                rejected_bbox[2:] += rejected_bbox[:2]
                
                x1,y1,x2,y2 = rejected_bbox.astype(np.int16)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                cv2.rectangle(frame, (x1, y1-30), (x1+(len(rejected_names[i])+len(str(rejected_scores[i])))*17, y1), color, -1)
                cv2.putText(frame, rejected_names[i]+'-'+str(rejected_scores[i]), (x1, y1-10), 0, 0.75, (0, 0, 0), 2)        
        
        # update tracks
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            x1,y1,x2,y2 = track.to_tlbr().astype(np.int16)
            class_name = track.get_class()
            
        # draw bbox on screen
            if track.matched:
                color = (255, 0, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.rectangle(frame, (x1, y1-30), ((x1)+(len("target"))*15, y1), color, -1)
                cv2.putText(frame, "[target]",(x1, y1-10),0, 0.75, (255,255,255),2)      
            elif FLAGS.plot_tracking:
                color = colors[int(track.track_id) % len(colors)]
                color = [i * 255 for i in color]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.rectangle(frame, (x1, y1-30), ((x1)+(len(class_name)+len(str(track.track_id)))*17, y1), color, -1)
                cv2.putText(frame, class_name + "-" + str(track.track_id),(x1, y1-10),0, 0.75, (255,255,255),2)      
                
            max_instance = max(max_instance, track.track_id)
        
        # show maximum instance
        string = f"max : {str(max_instance)}"
        cv2.putText(frame, string, (10, 30), 0, 0.75, (255, 255, 255), 2)
        
        
        # calculate frames per second of running detections
        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        if not FLAGS.dont_show:
            cv2.imshow("Output Video", result)
        
        # if output flag is set, save video file
        if FLAGS.output:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        if FLAGS.debug:
            break
    cv2.destroyAllWindows()       
    
    print(appear_in_frame)
    print(min_track_id, appear_in_frame[min_track_id])
    

if __name__=='__main__':
    try:
        app.run(main)
    except SystemExit:
        pass