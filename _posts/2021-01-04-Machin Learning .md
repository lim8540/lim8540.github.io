---
category: programmers
tags: [K-digital training, week5_day1, ml_basics]
use_math: true
---
 
## Machine Learning 시작하기

### Machine Learning 이란?
- 경험을 통해 자동으로 개선하는 컴퓨터 알고리즘의 연구
- 학습데이터 : 입력 벡터들 $x_1, ..., x_N,$ 목표값들 $t_1, ..., t_N$
- 머신러닝 알고리즘의 결과는 목표값을 예측하는 함수 $y(x)$

### 핵심개념들
- 학습단계(training or learning phase) : 함수 $y(x)$를 학습데이터에 기반해 결정하는 단계
- 시험셋(test set) : 모델을 평가하기 위해서 사용하는 새로운 데이터
- 일반화(generalization) : 모델에서 학습에 사용된 데이터가 아닌 이전에 접하지 못한 새로운 데이터에 대해 올바른 예측을 수행하는 역량
- 지도학습(supervised learning) : target이 주어진 경우
    - 분류(classification)
    - 회귀(regression)
- 비지도학습(unsupervised learning) : target이 없는 경우
    - 군집(clustering)

### 다항식 곡선 근사(Polynomial Curve Fitting)
- 학습데이터 : 입력벡터 $X = (x_1,...,x_N)^T, t = (t_1, ..., t_N)^T$
- 목표 : 새로운 입력벡터 $\hat x$가 주어졌을 때 목표값 $\hat t$를 예측하는 것
- 확률이론(probability theory) : 예측값의 불확실성을 정량화시켜 표현할 수 있는 수학적인 프레임워크를 제공한다.
- 결정이론(decision theory) : 확률적 표현을 바탕으로 최적의 예측을 수행할 수 있는 방법론을 제공한다.
- A polynomial function linear in w
    - $y(x, w) = w_0 + w_1x + w_2 x^2 + ... + w_Mw^M = \sum_{j=0}^Mw_jx^j$
    - w에 관해서는 선형식, x에 대해서는 다항식
- 오차함수(Error Function)
    - $E(w) = {1 \over 2} {\sum_{n=1}^N(y(x_n,w) - t_n)^2}$
    - 실제값과 함수를 통해 예측된 값의 차이의 크기
- 과소적합(under-fitting)과 과대적합(over-fitting)
    - 예측함수의 M값 즉, 차원을 정해주어야 한다. 적합한 M보다 작은 값을 취하는 경우는 과소적합, 그 반대의 경우는 과대적합의 상태라고 한다.
    - $E_{RMS} = \sqrt{2E(w^*)/N}$
    - 데이터가 많아질 수록 복잡한 모델도 과대적합 문제가 덜 심각해짐을 알 수 있다. 모든 머신러닝 프로젝트의 핵심은 많은 데이터의 수집이다.
- 규제화(Regularization)
    - $\tilde E(w) = {1 \over 2} {\sum_{n=1}^N(y(x_n,w) - t_n)^2} + {\lambda \over 2} {\Vert w \Vert}^2$
    - where ${\Vert w \Vert}^2 \equiv w^Tw = w_0^2 + w_1^2 + ... + w_M^2$ 
    - 수식의 뒷부분 ${\lambda \over 2} {\Vert w \Vert}^2$ 이 규제화를 의미함. 람다값이 커질 수록 w값의 절대값을 작게 만드는 모델로 가게됨(규제화에 신경을 씀). 이에 대한 tradeoff는 에러를 해결하지 못하게 되는 것.