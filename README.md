- [Bigstar 2022](#bigstar-2022)
  - [수행 과제 개요](#수행-과제-개요)
  - [Problem Definition](#problem-definition)
  - [Data Augmentation](#data-augmentation)
    - [Data Space 증강](#data-space-증강)
    - [Feature Space 증강](#feature-space-증강)
  - [접근 방법](#접근-방법)
    - [Easy Data Augmentation](#easy-data-augmentation)
    - [Backtranslation](#backtranslation)
    - [Generative Approach](#generative-approach)
    - [Comparison](#comparison)
  - [Implementation Detail](#implementation-detail)
    - [Fine-tuning](#fine-tuning)
    - [Generating Data](#generating-data)
    - [Data Filtering](#data-filtering)
  - [Dataset](#dataset)
    - [Stats](#stats)
  - [Experimental Results](#experimental-results)


# Bigstar 2022

## 수행 과제 개요

![fig1](https://user-images.githubusercontent.com/7765506/227816649-01a1d337-7c3e-4d9d-ae0b-338be4342e1a.jpg)

=> 자연어 인공지능 모델 성능 향상을 위한 텍스트 데이터 증강 모델 개발.

## Problem Definition

![fig2](https://user-images.githubusercontent.com/7765506/227818426-61599ed5-f3e5-4b33-9687-9ef470a8db3c.jpg)

- 머신러닝 모델의 일반화 성능을 높이기 위해서는 많은 양의 데이터가 요구됨.
- 하지만, 다량의 데이터를 수집하고 annotation하는 데에는 많은 비용이 소모됨.

## Data Augmentation

- 적은 비용으로 좋은 모델을 만들기 위해 데이터 증강이 활용될 수 있음.
- 자연어 task에서 데이터를 증강하는 방법은 크게 2가지 존재.
  - Data space에서 수행되는 데이터 증강.
  - Feature space에서 수행되는 데이터 증강.

### Data Space 증강

![fig3](https://user-images.githubusercontent.com/7765506/227820164-8c0eecc7-569a-4682-bae0-cf91e4127140.jpg)

- 텍스트 형태의 data를 그대로 생성하는 방법.

### Feature Space 증강

![fig4](https://user-images.githubusercontent.com/7765506/227824458-c07cb5b3-83d0-44ab-af2c-ba810fd22e58.jpg)

- Feature space (latent space) 에서 수행되는 증강 기법.

## 접근 방법

- 다양한 모델에서도 사용할 수 있는 Data Space 증강 선택.
  - Easy Data Augmentation
  - Backtranslation
  - Generative Approach

### Easy Data Augmentation

- SR: Synonym Replacement (유의어 교체)
- RI: Random Insertion (랜덤 삽입)
- RS: Random Swap (랜덤 스왑)
- RD: Random Deletion (랜덤 삭제)

![table1](https://user-images.githubusercontent.com/7765506/231036943-5c706fcc-1fb8-492f-b17e-447097101528.jpg)

Pros: 4개의 간단한 operation으로 가장 빠르게 많은 데이터를 생성할 수 있다는 장점을 가짐.  
Cons: 자연스러운 문장 구조로 생성되지 못하고, 기존의 의미와 다른 의미를 가진 문장이 생성될 수 있음.

### Backtranslation

- 입력 언어에서 다른 언어로 번역 후 다시 원래 입력 언어로 번역하여 데이터를 증강하는 방법.


![fig5](https://user-images.githubusercontent.com/7765506/231367280-bda82af4-e942-45c9-b193-98ac1fac1218.jpg)

Pros: 자연스러운 문장 생성이 가능함.  
Cons: 다양한 문장이 생성되기 어려움.


### Generative Approach

- Pretrained Language Model(PLM)을 이용하여 문장을 생성하는 방법.

![fig6](https://user-images.githubusercontent.com/7765506/233258022-b2f24c86-7bc9-4ac7-bdd7-ba1d6f497fe6.jpg)

Pros: 자연스러운 문장 생성이 가능함.  
Cons: 위의 2개 방법과 달리 좋은 PLM은 계산 복잡도가 높으므로 문장 생성 속도가 상대적으로 느림.

### Comparison

![table2](https://user-images.githubusercontent.com/7765506/233268164-8819912d-6d35-454e-a9cb-7cde6f3367ac.jpg)

- 3가지 실험 결과 중 Generative Approach가 가장 유의미한 성능 향상 효과를 보임.

## Implementation Detail

- Pretrained Model
  - [skt/kogpt2-base-v2](https://huggingface.co/skt/kogpt2-base-v2)
- Fine-tuning
  - 주어진 데이터와 비슷한 문장을 생성하기 위해 find-tuning 진행.
  - 주어진 training set을 이용하여 auto-regression 학습을 진행.
- Generating Data
  - 문장을 원하는 수만큼 생성.
- Data Filtering
  - 생성된 문장 중에서 특정 기준에 부합하는 문장만 필터링.

### Fine-tuning

- Label invarint 문장을 생성하기 위해 다음과 같은 형태로 training set을 재구성.
  - $y_1SEPx_1EOSy_2SEPx_2EOSy_3\cdots y_nSEPx_nEOS$

![fig7](https://user-images.githubusercontent.com/7765506/233274513-b58ee384-a83e-42cd-ba13-c4fee95b2e1a.jpg)

- Auto-regressive 방식으로 학습 진행.

![fig8](https://user-images.githubusercontent.com/7765506/233276583-d7aa6e90-c8fe-496f-bee6-fa8a946c6236.jpg)

### Generating Data

- \<Label>\<SEP> 형식으로 모델의 input으로 넣어줌.

![fig9](https://user-images.githubusercontent.com/7765506/233277134-ab3377b0-6d1e-4b3f-b44d-a3b7e0f4bae0.jpg)

### Data Filtering

- 생성된 문장의 퀄리티를 보장하기 위해 특정 기준에 부합하는 문장만 사용.
  - 기존 training set $D_{train}$을 이용하여 $classifier$를 학습.
  - 학습된 $classifier$를 이용하여 생성한 $D_{aug}$에 대해 prediction 진행.
  - Ground truth와 prediction이 일치하는 문장만 필터링.

![fig10](https://user-images.githubusercontent.com/7765506/233279831-3a346f61-c5cf-466d-98e8-55a6c3d24506.jpg)

## Dataset

[소상공인 고객 주문 질의응답 텍스트](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=102)

| Column  | Description                    |
|---------|--------------------------------|
| IDX     | 고유 번호                      |
| 발화자  | 발화자 정보 (c: 고객, s: 점원) |
| 발화문  | 발화 텍스트 데이터             |
| QA 여부 | 질의문 (q), 응답문 (a) 여부    |
| 인텐트  | 발화문에 내재된 의도           |
| ...    |            ...         |

### Stats

| $N_{train}$ | $N_{val}$ | $N_{test}$ | $N_{class}$ |
|-------------|-----------|------------|-------------|
| 652,994     | 84,814    | 108,720    | 13          |

![fig11](https://user-images.githubusercontent.com/7765506/234186199-395ed70f-1feb-4c35-9413-aed12ef5ffb9.jpg)


## Experimental Results

![fig13](https://user-images.githubusercontent.com/7765506/234771730-705d26c9-a899-45a0-b8ed-5a9650fe5ec1.jpg)
