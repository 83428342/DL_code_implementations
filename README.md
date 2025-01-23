# 밑바닥부터 시작하는 딥러닝

이 프로젝트는 '밑바닥부터 시작하는 딥러닝' 시리즈를 참고하여 기본적인 딥러닝 코드를 구현하는 연습을 위한 저장소입니다.

## 폴더 구조

```
src/
├── data/              # 데이터 로딩 및 전처리 관련 코드
│   ├── __init__.py
│   └── mnist.py       # MNIST 데이터 불러오기
├── layers/            # 모델 구성 레이어 정의
│   ├── __init__.py
│   └── activations.py # 활성화 함수 정의
├── losses/            # 손실 함수 정의 
│   └── __init__.py
├── models/            # 모델 아키텍처 정의
│   ├── __init__.py
│   └── model1.py      # 3층 layer 정의
├── utils/             # 유틸리티 함수 및 클래스
│   └── __init__.py
└── README.md         
```

## 1권

*   activation function
*   Mnist를 이용한 코드 구현
