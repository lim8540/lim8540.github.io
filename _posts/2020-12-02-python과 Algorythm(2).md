---
category: programmers
tags: [K-digital training, week1_day3, algorithms]
---

## 가장 큰 수 (조건에 따라 sort 하기)

문제: [Programmers](https://programmers.co.kr/learn/courses/30/lessons/42746)


1. 기존의 구현(함수를 통한 sort)
```python
from functools import cmp_to_key

def compare(x, y):
    
    tmp1 = str(x) + str(y)
    tmp2 = str(y) + str(x)
    if(tmp1 > tmp2):
        return -1
    elif(tmp1 < tmp2):
        return 1
    else:
        return 0


def solution(numbers):
    answer = ''
    tmp = sorted(numbers, key = cmp_to_key(compare))
    for i in range(len(tmp)):
        tmpstr = str(tmp[i])
        answer += tmpstr
    while(len(answer) > 1):
        if(int(answer[0]) == 0):
            answer = answer[1:]
        else:
            return answer
    return answer
```

functools library에 cmp_to_key를 이용하였다. 이게 예전 python에는 있던 내용인데 빠졌다고 한다. compare함수를 만들어서 두 값을 서로 반대로 이어 붙여 보아서 큰 수가 나오는 순으로 정렬하도록 하였다.

2. 강의에서의 구현

위처럼 library를 쓰지 않고 key를 lambda를 통해 지정해 주는 방법을 전에도 고민을 했었는데 생각이 나지 않아서 위처럼 문제를 풀었었다. 이번 강의를 통해 위의 조건을 만족하도록 하는 방법을 알게 되었다. 두 수를 두번 이어붙이지 않고 4칸까지 채운후 비교하는 방법이 있었다. 예를들어 34와 343을 비교할 때 34를 이어늘리면 34343434...가 되고 343을 이어붙이면 343343343...이 된다. 이걸 네자리까지 비교하면(입력받는 수의 최대 자리수가 4자리 이므로) 34가 더 앞에 오게 된다. 1번의 방법을 쓰면 34343 vs 34334가 되기 때문에 34가 343보다 앞에 오게된다. 구현은 아래와 같다.
```python
def solution(numbers):
    numbers = [str(x) for x in numbers]
    numbers.sort(key = lambda x : (x * 4)[:4], reverse=True)
    answer = ''.join(numbers)
    while answer != '' and answer[0] == '0':
        answer = answer[1:]
    if answer == '':
        answer = '0'
    return answer
```

## dictionary에 default값 주기
```python

def solution(participant, completion):
    answer = ''
    d = {}
    for x in participant:
        d[x] = d.get(x, 0) + 1
    for x in completion:
        d[x] -= 1
    dnf = [k for k, v in d.items() if v > 0]
    answer = ''.join(dnf)
    return answer

```
위의 코드에서 값을 하나 더하는 것과 초기값 세팅을 해주는 부분이 아래의 부분이다.
```python
    d[x] = d.get(x,0) + 1
```
get(x,0)에서 x에 해당하는 값이 없을 때에는 0으로 초기화해주고 있을때는 +1을 해주게 된다.   
처음 배웠을 때는 이해했다 하고 넘어갔는데 다시 코드로 구현해보려니까 생각이 나지 않아서 여기에 적었다.

