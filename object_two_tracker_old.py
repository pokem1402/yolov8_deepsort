import time
from absl.flags import FLAGS
from absl import app, flags
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchvision
import torch

from utils import util

# reid
from reid import generate_detection_fast_reid as gdet

# yolo v8
from ultralytics import YOLO

# deep sort
from deep_sort import nn_matching, preprocessing
from deep_sort.tracker import Tracker
from deep_sort.detection import Detection

flags.DEFINE_string("weights_path", "./weights/", "path to weights files")
flags.DEFINE_string("sort_weights", "reid/VeRi_dynamic.onnx", "deep feature extractor weights")
flags.DEFINE_string("reid_weights", None, "ReID pretrained model")
flags.DEFINE_string("weights", 'yolov8m.pt', "path to weights file")
flags.DEFINE_string("video1", "./data/section_cam.mp4", "path to input video1 or set to 0 for webcam")
flags.DEFINE_string("video2", "./data/number_recog.mp4", "path to input video2")
flags.DEFINE_string("output", None, "path to output video")
flags.DEFINE_string("output_format", "XVID", "codec used in VideoWriter when savinng video to file")
flags.DEFINE_integer('size', 640, "resize images to")
flags.DEFINE_float("iou", 0.5, "iou threshold")
flags.DEFINE_float("conf", 0.4, "score threshold") # 0.5
flags.DEFINE_boolean("dont_show", False, "dont_show video output")
flags.DEFINE_boolean("plot_tracking", False, "True if don't want tracking non-matched car ")
flags.DEFINE_string("video_mask1", "masking_video1.jpg", "path to input video mask")
flags.DEFINE_string("video_mask2", "masking_video2.jpg", "path to input video mask2")
flags.DEFINE_float("max_cosine_distance", 0.4, "cosine distance threshold")
flags.DEFINE_float("frame_skip", 0.85, "frame skip rate")
flags.DEFINE_float("object_ignore1", 0.03, "ignore a object if the object size smaller than frame_height * value")
flags.DEFINE_float("upper_limit1", 0., "ignore a object if center of the object in upper limit")
flags.DEFINE_float("lower_limit1", 0.2, "ignore a object if center of the object in lower limit") # 0.
flags.DEFINE_float("object_ignore2", 0.05, "ignore a object if the object size smaller than frame_height * value") # 0.3
flags.DEFINE_float("upper_limit2", 0.4, "ignore a object if center of the object in upper limit") # 0.2
flags.DEFINE_float("lower_limit2", 0.4, "ignore a object if center of the object in lower limit") # 0.3
flags.DEFINE_float("min_cosine_distance", 0.73, "unmatch if distance greater than this value") # 0.7
flags.DEFINE_boolean("debug", False, "for debug")
def main(_argv):
    
    # parameters
    max_cosine_distance = FLAGS.max_cosine_distance
    nms_max_overlap = 1.0
    nn_budget = None
    
    # initialize deep sort
    sort_model_filename = FLAGS.weights_path + FLAGS.sort_weights
    encoder_sort = gdet.create_box_encoder(sort_model_filename, batch_size=1)
    
    if FLAGS.reid_weights:
        reid_model_filename = FLAGS.weights_path + FLAGS.reid_weights
        encoder_reid = gdet.create_box_encoder(reid_model_filename)
    else:
        encoder_reid = encoder_sort
    
    # calculate cosine distance metric
    metric1 = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    metric2 = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    
    
    # initialize tracker
    tracker1 = Tracker(metric1, 0.8)
    tracker2 = Tracker(metric2, 0.8)
    tracker = [tracker1, tracker2]
    
    video_path1 = FLAGS.video1
    video_path2 = FLAGS.video2
    
    input_size = FLAGS.size
    
    # load yolo model
    yolo = YOLO(FLAGS.weights_path + FLAGS.weights)
    
    # begin video capture
    try:
        vid1 = cv2.VideoCapture(int(video_path1))
    except:
        vid1 = cv2.VideoCapture(video_path1)
    
    vid2 = cv2.VideoCapture(video_path2)
    
    # get video mask if needed
    if FLAGS.video_mask1:
        mask1 = cv2.imread(FLAGS.video_mask1)//255
    
    if FLAGS.video_mask2:
        mask2 = cv2.imread(FLAGS.video_mask2)//255
    
    
    out = None
    width1 = int(vid1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height1 = int(vid1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    width2 = int(vid2.get(cv2.CAP_PROP_FRAME_WIDTH))
    height2 = int(vid2.get(cv2.CAP_PROP_FRAME_HEIGHT))

    width = [width1, width2]
    height = [height1, height2]
    
    # get video ready to save locally if flag is set
    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        fps = int(vid1.get(cv2.CAP_PROP_FPS)*(1.0 - FLAGS.frame_skip))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width1, height1))
    
    last_frame_num1 = vid1.get(cv2.CAP_PROP_FRAME_COUNT)
    last_frame_num2 = vid2.get(cv2.CAP_PROP_FRAME_COUNT)
    
    total_frame1 = int(last_frame_num1 *(1.0 - FLAGS.frame_skip))
    total_frame2 = int(last_frame_num2 *(1.0 - FLAGS.frame_skip))
    
    total_frame = min(total_frame1, total_frame2)
    per_frame1 = [int((last_frame_num1 / total_frame)),int(round(last_frame_num1 / total_frame))]
    per_frame2 = [int((last_frame_num2 / total_frame)), int(round(last_frame_num2 / total_frame))]

    print(last_frame_num1 / total_frame, last_frame_num2 / total_frame)

    upper_limit = [FLAGS.upper_limit1, FLAGS.upper_limit2]
    lower_limit = [FLAGS.lower_limit1, FLAGS.lower_limit2]
    object_ignore = [FLAGS.object_ignore1, FLAGS.object_ignore2]

    # initialize color map
    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]
    
    
    match_id = -1
    progress = 0
    
    max_instance = [0, 0]
    # while video is running
    for _ in tqdm(range(total_frame), desc="Frame #"):
        
        success1, frame1 = vid1.read()
        success2, frame2 = vid2.read()
                  
        for _ in range(per_frame1[progress%2]-1):
            success, _ = vid1.read()
            if not success:
                print("Video1 has ended or failed")
                break
        for _ in range(per_frame2[progress%2]-1):
            success, _ = vid2.read()
            if not success:
                print("Video2 has ended or failed")
                break
        progress += 1
        
        if success1 and success2:
            frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
            frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
        else:
            print("One of Videos has ended or failed")
            break
        
        if FLAGS.video_mask1:
            image_data1 = cv2.resize(frame1 * mask1, (input_size, input_size))
        else:
            image_data1 = cv2.resize(frame1, (input_size, input_size))
        if FLAGS.video_mask2:
            image_data2 = cv2.resize(frame2 * mask2, (input_size, input_size))
        else:
            image_data2 = cv2.resize(frame2, (input_size, input_size))

        frame = [frame1, frame2]
        image_data = [image_data1, image_data2]
        
        for i in range(2):
                    
            # yolo
            results = yolo(image_data[i], iou = FLAGS.iou, conf = FLAGS.conf, verbose=False)[0]
            
            # run nms
            indices = torchvision.ops.nms(results.boxes.xyxy, results.boxes.conf, FLAGS.iou).cpu().numpy()
            
            factor = np.array([width[i], height[i], width[i], height[i]])[np.newaxis, ...]
            boxes = results.boxes.xyxyn.cpu().numpy()        
            boxes[:, 2:] = results.boxes.xywhn[:, 2:].cpu().numpy() 
            boxes = (boxes*factor).astype(np.uint16)[indices]
            scores = results.boxes.conf[indices]
            classes = results.boxes.cls[indices]

            # relative position filtering        
            accepted, _ = util.region_filtering_by_relative_position(boxes, height[i],
                                                                    upper_limit=upper_limit[i],
                                                                    lower_limit=lower_limit[i])

            boxes = boxes[accepted]
            scores = scores[accepted]
            classes = classes[accepted]
            
            # object filtering       
            accepted, _ = util.object_filtering(boxes, width[i], height[i], object_ignore[i])

            boxes = boxes[accepted]
            scores = scores[accepted]
            classes = classes[accepted]
           
            # read all class name
            class_names = util.read_class_name(f"./utils/coco.names")
            
            allowed_classes = ['car', 'truck']
            
            names = []
            indx = []
            for j in range(len(classes)):
                class_indx = int(classes[j])
                class_name = class_names[class_indx]
                if class_name in allowed_classes:
                    indx.append(j)
                    names.append('car')
            bboxes = boxes[indx]
            scores = scores[indx]

            # encode yolo detections and feed to tracker
            
            features = encoder_sort(frame[i], bboxes)
            detections = [Detection(bbox, score, class_name, feature)
                        for bbox, score, class_name, feature
                        in zip(bboxes, scores, names, features)]       
        
            # run nms
            boxs = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            classes = np.array([d.class_name for d in detections])
            indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
            detections = [detections[j] for j in indices] 
            
            # Call the tracker
            tracker[i].predict()
            tracker[i].update(detections)
    

        query_idx = -1
        query_size = 0
        query_bbox = None
        query_feat = None
        gallery_idxs = []
        gallery_feat = None
        gallery_bbox = []
        gallery_match_ids= []
        
        for i, query in enumerate(tracker[1].tracks):
            
            if not query.is_confirmed() or query.time_since_update > 1:
                continue
            x,y,w,h = query.to_tlwh()
            
            if w*h > query_size:
                query_idx = i
                query_size = w*h
                query_bbox = query.to_tlwh()[np.newaxis, ...]
            
        for i, gallery in enumerate(tracker[0].tracks):
            if not gallery.is_confirmed() or gallery.time_since_update > 1:
                continue
            gallery_idxs.append(i)
            gallery_bbox.append(gallery.to_tlwh())
            gallery_match_ids.append(gallery.match_id)
    
        # print(query_match_id)
        # print(gallery_match_ids)
    
        # if (query_match_id == -1) or (query_match_id not in gallery_match_ids):
               
        if query_bbox is not None and gallery_bbox:
            gallery_bbox = np.stack(gallery_bbox)
            query_feat = encoder_reid(frame2, query_bbox)
            gallery_feat = encoder_reid(frame1, gallery_bbox)
        
            cosine_distance = nn_matching._cosine_distance(query_feat, gallery_feat)
            argmin = np.argmin(cosine_distance)
            gallery_idx = gallery_idxs[argmin]
            if cosine_distance[:, argmin] < FLAGS.min_cosine_distance: #TODO : 두 번째 re-id 부터는 distance 허들을 더 높이자
                
                if tracker[1].tracks[query_idx].matched:
                    
                    query_id = tracker[1].tracks[query_idx].match_id
                    
                    for i, _match_id in enumerate(gallery_match_ids):
                        if _match_id == query_id:
                            tracker[0].tracks[gallery_idxs[i]].matched= False
                            tracker[0].tracks[gallery_idxs[i]].match_id= -1                           
                            
                    tracker[0].tracks[gallery_idx].matched = True
                    tracker[0].tracks[gallery_idx].match_id = query_id
                else:
                    match_id += 1
                    # if tracker[0].tracks[gallery_idx].matched:                        
                    #     tracker[1].tracks[query_idx].match_id = tracker[0].tracks[gallery_idx].match_id
                    #     tracker[1].tracks[query_idx].matched = True
                    # else:
                    tracker[1].tracks[query_idx].matched = True
                    tracker[1].tracks[query_idx].match_id = match_id
                    tracker[0].tracks[gallery_idx].matched = True
                    tracker[0].tracks[gallery_idx].match_id = match_id
                
        # update tracks
        for track in tracker1.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            x1,y1,x2,y2 = track.to_tlbr().astype(np.int16)
            class_name = track.get_class()
        
            if track.matched:
                color = colors[int(track.match_id) % len(colors)]
                color = [i * 255 for i in color]
                string = f"[car : {str(track.match_id)}]"
                cv2.rectangle(frame1, (x1, y1), (x2, y2), color, 2)
                cv2.rectangle(frame1, (x1, y1-30), ((x1)+(len(string)*13), y1), color, -1)
                cv2.putText(frame1, string,(x1, y1-10),0, 0.75, (255,255,255),2)          
            elif FLAGS.plot_tracking:
        # draw bbox on screen
                color = colors[int(track.track_id) % len(colors)]
                color = [i * 255 for i in color]
                cv2.rectangle(frame1, (x1, y1), (x2, y2), color, 2)
                cv2.rectangle(frame1, (x1, y1-30), ((x1)+(len(class_name)+len(str(track.track_id)))*17, y1), color, -1)
                cv2.putText(frame1, class_name + "-" + str(track.track_id),(x1, y1-10),0, 0.75, (255,255,255),2)      
            max_instance[0] = max(max_instance[0], track.track_id)
        
        # show maximum instance
        string = f"max : {str(max_instance[0])}"
        cv2.putText(frame1, string, (10, 30), 0, 0.75, (255, 255, 255), 2)
    
        frame2_resized = cv2.resize(frame2, (int(width2//4), int(height2//4)))

        # update tracks
        for track in tracker2.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            x1,y1,x2,y2 = track.to_tlbr().astype(np.int16)
            class_name = track.get_class()
            
            if track.matched:
                color = colors[int(track.match_id) % len(colors)]
                color = [i * 255 for i in color]
                string = f"[car : {str(track.match_id)}]"
                cv2.rectangle(frame2_resized, (x1//4, y1//4), (x2//4, y2//4), color, 2)
                cv2.rectangle(frame2_resized, (x1//4, y1//4-30), ((x1//4)+(len(string))*13, y1//4), color, -1)
                cv2.putText(frame2_resized, string,(x1//4, y1//4-10),0, 0.75, (255,255,255),2)                          
            elif FLAGS.plot_tracking:
        # draw bbox on screen
                color = colors[int(track.track_id) % len(colors)]
                color = [i * 255 for i in color]
                cv2.rectangle(frame2_resized, (x1//4, y1//4), (x2//4, y2//4), color, 2)
                cv2.rectangle(frame2_resized, (x1//4, y1//4-30), ((x1//4)+(len(class_name)+len(str(track.track_id)))*17, y1//4), color, -1)
                cv2.putText(frame2_resized, class_name + "-" + str(track.track_id),(x1//4, y1//4-10),0, 0.75, (255,255,255),2)      
            max_instance[1] = max(max_instance[1], track.track_id)
        
        # show maximum instance
        string = f"max : {str(max_instance[1])}"
        cv2.putText(frame1, string, (int(width[0]-width[1]//4)+ 10, int(height[1]//4)+30), 0, 0.75, (255, 255, 255), 2)
        
        frame_total = frame1.copy()
        frame_total[:int(height[1]//4), int(width[0]-width[1]//4):, :] = frame2_resized

        # calculate frames per second of running detections
        result = np.asarray(frame_total)
        result = cv2.cvtColor(frame_total, cv2.COLOR_RGB2BGR)
        
        
        if not FLAGS.dont_show:
            cv2.imshow("Output Video", result)
        
        # if output flag is set, save video file
        if FLAGS.output:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        if FLAGS.debug:
            break
    cv2.destroyAllWindows()       
        
        

if __name__=='__main__':
    try:
        app.run(main)
    except SystemExit:
        pass