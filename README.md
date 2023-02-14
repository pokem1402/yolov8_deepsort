# yolov8_deepsort


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
  
```
