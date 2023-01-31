import time
from absl.flags import FLAGS
from absl import app, flags
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchvision

from utils import util

# reid
from reid import generate_detection_fast_reid as gdet

# yolo v8
from ultralytics import YOLO

# deep sort
from deep_sort import nn_matching, preprocessing
from deep_sort.tracker import Tracker
from deep_sort.detection import Detection


flags.DEFINE_string("weights", './weights/yolov8m.pt', "path to weights file")
flags.DEFINE_string("video", "./data/cars.mp4", "path to input video or set to 0 for webcam")
flags.DEFINE_string("output", None, "path to output video")
flags.DEFINE_string("output_format", "XVID", "codec used in VideoWriter when savinng video to file")
flags.DEFINE_integer('size', 416, "resize images to")
flags.DEFINE_float("iou", 0.5, "iou threshold")
flags.DEFINE_float("conf", 0.5, "score threshold")
flags.DEFINE_boolean("dont_show", False, "dont_show video output")
flags.DEFINE_string("video_mask", None, "path to input video mask")
flags.DEFINE_float("max_cosine_distance", 0.4, "cosine distance threshold")
flags.DEFINE_string("class_name", "coco.names", "class name for model")

def main(_argv):
    
    # parameters
    max_cosine_distance = FLAGS.max_cosine_distance
    nn_budget = None
    nms_max_overlap = 1.0
    
    # initialize deep sort
    model_filename = "./weights/reid/veriwild_dynamic.onnx"
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    
    # initialize tracker
    tracker = Tracker(metric)
    
    video_path = FLAGS.video
    input_size = FLAGS.size
    
    # load yolo model
    yolo = YOLO(FLAGS.weights)
    
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
    
    # get video ready to save locally if flag is set
    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))
    
    last_frame_num = vid.get(cv2.CAP_PROP_FRAME_COUNT)
    
    # while video is running
    for frame_num in tqdm(range(int(last_frame_num)), desc="Frame #"):
        success, frame = vid.read()
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
        
        # indices = torchvision.ops.nms(results.boxes.xyxy, results.boxes.conf, FLAGS.iou).cpu().numpy()
        
        factor = np.array([width, height, width, height])[np.newaxis, ...]
       
        boxes = results.boxes.xyxyn.cpu().numpy()
        boxes[:, 2:] = results.boxes.xywhn[:, 2:].cpu().numpy()

        boxes = (boxes * factor).astype(np.uint16)
        scores = results.boxes.conf
        classes = results.boxes.cls
      
        # boxes = (boxes * factor).astype(np.uint16)[indices]
        # scores = results.boxes.conf[indices]
        # classes = results.boxes.cls[indices]
        
        # indices = torchvision.ops.nms(boxes, scores, FLAGS.iou)
        # print(indices)
        # break
        
        # read all class name
        class_names = util.read_class_name(f"./utils/{FLAGS.class_name}")
        
        allowed_classes = ['car']
        
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

        # encode yolo detections and feed to tracker
        
        features = encoder(frame, bboxes)
        detections = [Detection(bbox, score, class_name, feature)
                      for bbox, score, class_name, feature
                      in zip(bboxes, scores, names, features)]
        
        # initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]
        
        # # run nms
        # boxs = np.array([d.tlwh for d in detections])
        # scores = np.array([d.confidence for d in detections])
        # classes = np.array([d.class_name for d in detections])
        # indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        # detections = [detections[i] for i in indices] 
        
        # Call the tracker
        tracker.predict()
        tracker.update(detections)
        
        # update tracks
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            class_name = track.get_class()
            
        # draw bbox on screen
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
            cv2.putText(frame, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)      
        
         # calculate frames per second of running detections
        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        if not FLAGS.dont_show:
            cv2.imshow("Output Video", result)
        
        # if output flag is set, save video file
        if FLAGS.output:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cv2.destroyAllWindows()       
        
        

if __name__=='__main__':
    try:
        app.run(main)
    except SystemExit:
        pass