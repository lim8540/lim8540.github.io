---
category: programmers
tags: [K-digital training, week1_day4]
---

## 더 맵게(python에서 heap 사용하기)

문제: [Programmers](https://programmers.co.kr/learn/courses/30/lessons/42626)

```python

import heapq

def solution(scoville, K):
    answer = 0
    # 리스트 scoville로 부터 min heap구성
    heapq.heapify(scoville)
    while True:
        # min heap scoville 에서 최소값 삭제(반환)
        min1 = heapq.heappop(scoville)
        if min1 >= K:
            break
        elif len(scoville) == 0:
            answer = -1
            break
        min2 = heapq.heappop(scoville)
        new_scoville = min1 + 2 * min2
        # min heap scoville 에 원소 new_scoville 삽입
        heapq.heappush(scoville, new_scoville)
        answer += 1
    return answer

```

python에서 heap의 사용은 heapq library를 import해서 사용할 수 있다. 위에서 보이는 것처럼 list를 힙으로 바꾸어 줄 때는 다음과 같으 heapify를 사용한다.
```python
heapq.heapify(L)
```
heap에서 자료를 빼내고 삽입할때는 각각 heappop, heappush를 써준다. heap L에서 가장 작은 값을 빼주는 방법과 L에 새로운 원소 new_item을 삽입하는 방법은 아래와 같다.
```python
m = heapq.heappop(L)
heaqq.heappush(L, new_item)
```

## python의 zip
```python
A = [a1, a2, a3, a4]
B = [b1, b2 ,b3, b4]
for x, y in zip(A, B): #[(a1,b1), (a2, b2), (a3, b3)]
...
```
여러개의 리스트에서 동시에 원소를 뽑아낼 때 사용함.