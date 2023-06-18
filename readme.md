
# TEXT TO MBTI(Backend)
## 주제
텍스트를 입력하면 MBTI로 변환하여 여러 기능을 수행하는 프로그램

## 제작 기간
2023/03~2023/06

2023년 1학기 선문대학교 모바일프로그래밍 수업에서 제작한 프로그램


## 참여자
장효택,김연희,남영빈

## 사용 기술 및 도구
 Python-flask(api 구현)
 Tensorflow&Keras(mbti 모델 학습)

## 데이터셋


## 코드 설명
(use)만 볼것...
model(use).ipynb
 데이터셋 [Kaggle/mbti-personality-types-500-dataset](https://www.kaggle.com/datasets/zeyadkhalid/mbti-personality-types-500-dataset)을 사용하여 데이터를 학습시켜 H5모델로 변환하는 코드

predict_api.py
생성된 모델을 사용하여 api를 통해 웹서버로 모델        분석결과를 송출하는 코드