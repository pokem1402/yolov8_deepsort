# yolov8_deepsort + Vehicle Re-Identification between two videos


[Yolo v4 DeepSort](https://github.com/theAIGuysCode/yolov4-deepsort) 코드를 기반으로 [yolo v8](https://github.com/ultralytics/ultralytics) 모델을 사용하여 구축한 yolo deepsort입니다.

해당 코드를 만든 목적이 CCTV 영상에서의 차량 Tracking이기 때문에 Deep Sort의 Deep Appearance Descriptor는 [Fast-ReID](https://github.com/JDAI-CV/fast-reid)의 모델 중 veriwild-dataset으로 학습된 모델을 사용하였습니다.

## 구동 환경

모델의 실행을 위해서는 다음의 패키지를 설치해야합니다.
- [pytorch](https://pytorch.org/get-started/locally/) : Torch cuda version을 사용하고 싶은 경우에 먼저 설치하십시오.
- [ultralytics](https://github.com/ultralytics/ultralytics) : yolo v8 version에 대한 pip 패키지입니다.
```bash
    pip install ultralytics
```

## 실행

object_tracker.py 파일을 통해 object tracking을 수행할 수 있습니다.

```bash
object_tracker.py
  --weights : yolo 모델을 위한 weights 파일입니다. 모델에 대한 정보는 [공식 깃헙 Readme](https://github.com/ultralytics/ultralytics) 에서 확인 가능하고 모델명을 지정하면 자동으로 다운로드 받습니다.
  (default: ./weights/yolo8m.pt)
  --video : 입력할 비디오 스트림입니다. 0 입력 시 웹캠과 연결됩니다.
  (default: ./data/cars.mp4)
  --output : 출력할 비디오 위치입니다. 입력하지 않으면 저장하지 않습니다.
  (default: None)
  --output_format : 출력 비디오 포맷입니다.
  (default: XVID)
  --size : yolo가 사용하는 이미지 크기입니다.
  (default: 416)
  --iou : iou threshold
  (default: 0.5)
  --conf : confidence threshold
  (default: 0.5)
  --dont_show : 지정하면 출력하지 않습니다.
  (default: False)
  --video_mask : 비디오에 대한 화이트 리스트 구역을 지정하는 이미지 파일입니다.
  (default: None)
  --max_cosine_distance : cosine distance threshold
  (default: 0.4)
  --frame_skip : 비디오의 frame을 얼마나 skip할지 정합니다. [0, 1) 사이의 범위의 값으로 한정됩니다.
  (default: 0.0)
  --object_ignore : frame 대비 얼만큼 작은 물체들은 무시할지 정합니다 [0, 1) 사이의 범위의 값으로 한정됩니다.
  (default: 0.03)
  --upper_limit : 지정된 상단 영역에 물체의 중심 y 좌표가 놓인다면 무시합니다. [0, 1) 사이의 범위의 값으로 한정됩니다.
  (default: 0.)
  --lower_limit : 지정된 하단 영역에 물체의 중심 y 좌표가 놓인다면 무시합니다. [0, 1) 사이의 범위의 값으로 한정되고, upper_limit와 lower_limit의 합은 1 이하여야합니다.
  (default: 0.)
  
```

object_two_tracker.py 파일을 통해 두 영상간의 Re-identification을 수행할 수 있습니다.

```bash
object_two_tracker.py
  --weights_path : weight가 저장된 폴더입니다.
  (default: ./weights/)
  --weights : yolo 모델을 위한 weights 파일입니다. 모델에 대한 정보는 [공식 깃헙 Readme](https://github.com/ultralytics/ultralytics) 에서 확인 가능하고 모델명을 지정하면 자동으로 다운로드 받습니다.
  (default: yolo8m.pt)
  --sort_weights : DeepSORT에서 사용하는 Deep apearance descriptor를 지정합니다. onnx만 지원합니다.
  (default: reid/veriwild_dynamic.onnx)
  --reid_weights : 두 영상 간의 Re-Identification을 위해 사용하는 Re-ID모델을 지정합니다. onnx만 지원하고 지정하지 않으면 sort_weights와 동일한 것을 사용합니다.
  (default: None)
  --video1 : Gallery 영상이 될 영상 위치입니다. 0을 입력하면 웹캠도 지원합니다.
  (default: ./data/section_cam.mp4)
  --video2 : Query 영상이 될 영상 위치입니다.
  (default: ./data/number_recog.mp4)
  --output : 출력할 비디오 위치입니다. 입력하지 않으면 저장하지 않습니다.
  (default: None)
  --output_format : 출력 비디오 포맷입니다.
  (default: XVID)
  --size : yolo가 사용하는 이미지 크기입니다.
  (default: 640)
  --iou : iou threshold
  (default: 0.5)
  --conf : confidence threshold
  (default: 0.5)
  --dont_show : 지정하면 출력하지 않습니다.
  (default: False)
  --plot_tracking : 지정하면 matching 되지 않은 차량에 대한 추적을 bounding box로 그려줍니다.
  (default: False)
  --video_mask1 : Gallery 영상에 대한 video mask 이미지 위치입니다.
  (default: None)
  --video_mask2 : Query 영상에 대한 video mask 이미지 위치입니다.
  (default: None)
  --max_cosine_distance : cosine distance threshold
  (default: 0.4)
  --frame_skip : 비디오의 frame을 얼마나 skip할지 정합니다. [0, 1) 사이의 범위의 값으로 한정됩니다.
  (default: 0.85)
  --object_ignore1 : Gallary 영상의 frame 대비 얼만큼 작은 물체들은 무시할지 정합니다 [0, 1) 사이의 범위의 값으로 한정됩니다.
  (default: 0.03)
  --upper_limit1 : Gallary 영상의 지정된 상단 영역에 물체의 중심 y 좌표가 놓인다면 무시합니다. [0, 1) 사이의 범위의 값으로 한정됩니다.
  (default: 0.)
  --lower_limit1 : Gallary 영상의 지정된 하단 영역에 물체의 중심 y 좌표가 놓인다면 무시합니다. [0, 1) 사이의 범위의 값으로 한정되고, upper_limit와 lower_limit의 합은 1 이하여야합니다.
  (default: 0.2)
  --object_ignore2 : Query 영상 frame 대비 얼만큼 작은 물체들은 무시할지 정합니다 [0, 1) 사이의 범위의 값으로 한정됩니다.
  (default: 0.05)
  --upper_limit2 : Query 영상의 지정된 상단 영역에 물체의 중심 y 좌표가 놓인다면 무시합니다. [0, 1) 사이의 범위의 값으로 한정됩니다.
  (default: 0.4)
  --lower_limit : Query 영상의 지정된 하단 영역에 물체의 중심 y 좌표가 놓인다면 무시합니다. [0, 1) 사이의 범위의 값으로 한정되고, upper_limit와 lower_limit의 합은 1 이하여야합니다.
  (default: 0.4)
  --min_cosine_distance : 만약 query object와 cosine distance의 거리가 최소인 gallery object의 cosine distance가 이 값보다 크다면 같은 물체로 인식하지 않습니다.
  (default: 0.73)
```

## Result

Fast-ReID Pretrained model 간의 성능비교

![git](https://user-images.githubusercontent.com/18918072/218662959-a45b7405-6a07-4ddd-8538-0c02c5fa1571.gif)

더 긴 영상에 대해서는 [유투브 영상](https://youtu.be/sn4lJqrza5w)을 참조하실 수 있습니다.

## Data File

object_two_tracker.py 에서 사용한 데이터 파일은 [구글 드라이브](https://drive.google.com/drive/folders/1U-HkCiRLllAAYCbR0MkYH_QeLUld0Ony?usp=share_link)를 통해 다운로드 받을 수 있습니다.
