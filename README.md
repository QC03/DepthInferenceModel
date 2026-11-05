# Monocular Upper-Body Relative Depth

단일 RGB 영상에서 ResNet-18로 상체 키포인트의 허리 기준 상대 깊이를 회귀하는
학습용 프로젝트입니다. 목표는 센티미터 단위의 절대 깊이가 아니라 부위 간 전후
관계를 안정적으로 추정하는 것입니다.

## 데이터 준비

원본 이미지와 어노테이션은 다음 위치에 둡니다.

```text
data/raw_images/*.jpg
data/annotations.json
checkpoints/
```

샘플 영상은 1920x1080, 30fps이므로 먼저 프레임을 추출할 수 있습니다.

```powershell
New-Item -ItemType Directory -Force data\raw_images
ffmpeg -i resource\kimbumsoo_1.mp4 -vf "fps=5" data\raw_images\frame_%04d.jpg
```

ffmpeg가 없다면 포함된 OpenCV 유틸리티로 같은 작업을 할 수 있습니다.

```powershell
python extract_frames.py resource\kimbumsoo_1.mp4 --output-dir data\raw_images --every-n-frames 6
```

## 반자동 라벨링

MediaPipe Pose의 2D landmark와 `z_i-z_waist`를 어깨 간격으로 정규화해
`data/annotations.json`에 저장합니다. `z`는 MediaPipe 좌표계 부호를 그대로
저장하므로 값의 부호보다 허리 대비 상대 순서를 사용합니다. 가시성은 0.1~0.5로
보존하고, 양쪽 hip 기준점은 수동 GT와 같은 1.0으로 설정합니다.

```powershell
python labeling_assistant.py --input-dir data\raw_images
python labeling_assistant.py --input-dir data\raw_images --interactive
```

`--interactive`는 전체화면 라벨링 UI를 실행합니다. 왼쪽은 원본 이미지, 오른쪽
상단은 X-Y 스켈레톤, 오른쪽 하단은 Z-Y 상대 깊이 편집 화면입니다. `1~9`로
키포인트를 선택하고, 위/아래 방향키로 0.01씩 수정합니다. `w/s`는 0.1씩
수정하며 `Space` 또는 Enter로 저장 후 다음 이미지로 이동하고 `q`로 종료합니다.
수정한 값은 해당 키포인트의 confidence weight를 1.0으로 저장합니다.

## 학습

```powershell
python -m pip install -r requirements.txt
python train.py --annotations data\annotations.json --epochs 30
```

학습은 고정 seed(기본 42), 80/20 분할, AdamW(`1e-4`, `1e-4`), CosineAnnealingLR를
사용합니다. 손실은 confidence-weighted Smooth L1이며 검증 weighted MAE가 가장
낮은 모델을 `checkpoints/best_model.pth`에 저장합니다. OneDrive 동기화나 다른
프로그램이 파일을 잠시 점유해도 epoch별 파일을 먼저 저장하고 최대 5회 재시도합니다.

## 추론 및 시각화

이미지:

```powershell
python inference.py data\raw_images\frame_0001.jpg --checkpoint checkpoints\best_model.pth --output result.jpg
```

영상:

```powershell
python inference.py resource\kimbumsoo_1.mp4 --checkpoint checkpoints\best_model.pth
```

결과는 자동으로 `output\result_1.mp4`, `output\result_2.mp4`처럼 저장되며 기존
파일을 덮어쓰지 않습니다. 추론 UI는 시작 시 전체화면으로 열리고, 왼쪽에는
원본 영상, 오른쪽 상단에는 X-Y 스켈레톤, 오른쪽 하단에는 Z-Y 상대 깊이
스켈레톤을 표시합니다. 종료는 `q`입니다. 서버나 GUI가 없는 환경에서는
`--no-display`를 사용하면 화면 없이 동일한 결과를 저장합니다.

빨강 계열은 허리보다 상대적으로 앞, 파랑 계열은 뒤쪽인 값을 표시합니다.
추론 전용 checkpoint가 없으면 먼저 실제 촬영 데이터로 라벨링과 학습을 수행해야
합니다. ImageNet 가중치는 첫 모델 생성 시 torchvision 캐시에 다운로드됩니다.
