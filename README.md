# Monocular Upper-Body Relative Depth

단일 RGB 영상에서 ResNet-18로 상체 키포인트의 허리 기준 상대 깊이를 추정합니다.
절대 cm 깊이가 아니라 부위 간 전후 관계를 안정적으로 구분하는 것이 목표입니다.

## 사용법 타임라인

```powershell
# 1. 의존성 설치
python -m pip install -r requirements.txt

# 2. 샘플 영상에서 프레임 추출
python src\extract_frames.py resource\kimbumsoo_1.mp4 --output-dir data\raw_images --every-n-frames 6

# 3. MediaPipe 초기 라벨 생성
python src\labeling_assistant.py --input-dir data\raw_images

# 4. 전체화면 수동 라벨링
python src\labeling_assistant.py --input-dir data\raw_images --interactive

# 5. 학습: confidence loss + 관절별 3D 길이 min/max penalty
python src\train.py --annotations data\annotations.json --epochs 30

# 6. 전체화면 추론 및 자동 저장
python src\inference.py resource\kimbumsoo_1.mp4 --checkpoint checkpoints\best_model.pth
```

추론 결과는 `output\result_1.mp4`, `output\result_2.mp4`처럼 자동 저장됩니다.
소스 Python 파일은 모두 `src\` 폴더에 있습니다.

## 데이터 준비

```text
data/raw_images/*.jpg
data/annotations.json
checkpoints/
```

ffmpeg를 사용한다면 다음처럼 추출할 수도 있습니다.

```powershell
New-Item -ItemType Directory -Force data\raw_images
ffmpeg -i resource\kimbumsoo_1.mp4 -vf "fps=5" data\raw_images\frame_%04d.jpg
```

## 수동 라벨링 UI

```powershell
python src\labeling_assistant.py --input-dir data\raw_images --interactive
```

전체화면 UI는 왼쪽에 원본 이미지, 오른쪽 상단에 X-Y 스켈레톤, 오른쪽
하단에 Z-Y 상대 깊이 편집기를 표시합니다.

- `1~9`: 키포인트 선택
- 위/아래 방향키: 선택된 깊이 `+/-0.01`
- `w/s`: 선택된 깊이 `+/-0.10`
- `Space` 또는 `Enter`: 현재 라벨 저장 후 다음 이미지
- `q` 또는 `ESC`: 종료

수동으로 수정한 키포인트의 confidence weight는 `1.0`으로 저장됩니다.

## 학습과 관절 길이 제약

```powershell
python src\train.py --annotations data\annotations.json --epochs 30
```

학습은 고정 seed, 80/20 분할, AdamW(`lr=1e-4`, `weight_decay=1e-4`),
CosineAnnealingLR를 사용합니다. 기본 손실은 confidence-weighted Smooth L1입니다.
추가로 각 스켈레톤 연결의 2D 좌표와 예측 상대 깊이를 결합한 3D 길이를 계산합니다.
라벨 데이터의 연결별 최소·최대 길이에 15% 안전 여유를 둔 범위를 만들고, 예측
길이가 범위 밖일 때만 squared penalty를 더합니다. 이를 통해 관절 길이가 비현실적으로
짧거나 길어지는 것을 제한합니다.

검증 weighted MAE가 가장 낮은 모델을 `checkpoints\best_model.pth`에 저장합니다.
OneDrive 동기화나 다른 프로그램이 파일을 잠시 점유해도 epoch별 파일을 먼저
저장하고 최대 5회 재시도합니다.

## 추론 UI

```powershell
python src\inference.py resource\kimbumsoo_1.mp4 --checkpoint checkpoints\best_model.pth
```

추론 창은 전체화면으로 시작합니다.

- 왼쪽: 원본 영상
- 오른쪽 상단: X-Y 스켈레톤
- 오른쪽 하단: Z-Y 상대 깊이 스켈레톤
- `q`: 종료

결과 영상은 `output\result_N.mp4`에 저장되며 기존 파일은 덮어쓰지 않습니다.
GUI가 없는 서버에서는 다음처럼 화면 없이 같은 결과를 저장할 수 있습니다.

```powershell
python src\inference.py resource\kimbumsoo_1.mp4 --checkpoint checkpoints\best_model.pth --no-display
```
