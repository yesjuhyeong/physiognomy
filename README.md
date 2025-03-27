# 관상 분석 시스템 (Face Reading Application)

본 프로젝트는 얼굴 랜드마크 인식을 바탕으로, 현대적인 방식으로 관상을 해석하는 웹 서비스입니다.
카메라 혹은 업로드된 사진을 통해 얼굴을 인식하고, 주요 특징을 분석하여 눈 모양과 연관된 성격적 경향을 추정합니다.

## 기능

- 실시간 얼굴 캡처: 웹캠을 이용한 실시간 얼굴 촬영
- 이미지 파일 분석: 업로드된 이미지에 대한 얼굴 인식 및 분석
- 랜드마크 시각화: 68개 얼굴 특징점(랜드마크)을 시각적으로 표시
- 성격 예측: 관상학적 해석을 기반으로 한 성격적 특성 추정

## 설치 방법

1. 프로젝트 클론

    ```bash
    git clone https://github.com/yesjuhyeong/physiognomy.git
    cd physiognomy
    ```

2. 라이브러리 설치

    ```bash
    pip install -r requirements.txt
    ```

3. dlib 랜드마크 예측 모델 다운로드

- dlib의 68개 랜드마크 예측 모델 다운로드:

    ```bash
    # 다운로드
    curl -L -o shape_predictor_68_face_landmarks.dat.bz2 http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

    # 압축 해제
    bunzip2 shape_predictor_68_face_landmarks.dat.bz2
    ```

- Windows:
  http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

  - 압축을 해제한 후 `shape_predictor_68_face_landmarks.dat` 파일을 프로젝트 루트 디렉토리에 위치시킵니다.

## 실행 방법

```bash
python app.py
```

- 브라우저에서 `http://localhost:5000`주소로 접속하면 애플리케이션에 접근할 수 있습니다.

## 필요 사항

- Python 3.8 이상
- 웹캠(실시간 분석을 사용할 경우)
- 웹 브라우저

## 참고 사항

- 조명이 밝은 환경에서 사용할수록 인식 정확도가 높아집니다.
- 안경을 착용하면 분석 결과가 다소 왜곡될 수 있습니다.
- 본 프로그램은 재미와 실험을 위한 목적으로 제작되었으며, 과학적 근거에 입각한 정밀한 성격 진단 도구는 아님을 유의 바랍니다.