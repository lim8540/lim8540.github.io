---
category: programmers
tags: [K-digital training, week2_day1, algorithms]
---

## 게임 아이템(python에서 heap 사용하기)

### 문제 개요
healths 에는 캐릭터들의 체력이 들어오고 items에는 item마다 늘어나는 공격력과 깎이는 체력이 들어있다. 아이템을 장착하였을때 남아있는 체력이 100이상이어야 하며 캐릭터 하나당 착용 가능한 아이템은 하나 뿐이다. 공격력이 최대가 되게하는 착용 아이템들의 index를 오름차순으로 return하라   

### 입출력 예시

|healths|items|return|
|---|:---:|---:|
|[200,120,150]|	[[30,100],[500,30],[100,400]]	|[1,2]|
|[300,200,500]|	[[1000, 600], [400, 500], [300, 100]]	|[3]|

입출력 예 #2   

첫 번째, 두 번째 아이템을 사용하면 캐릭터의 체력이 100 미만이 됩니다. 따라서 세 번째 아이템만 사용할 수 있습니다.



```python

import heapq

def solution(scoville, K):
    import heapq

def solution(healths, items):
    answer = []
    heap = []
    #체력을 낮은 순으로 정렬한다.
    healths.sort()
    for i, item in enumerate(items):
        item.append(i+1)
    #items를 깎이는 채력이 낮은 순으로 정렬한다.
    items.sort(key = lambda x : x[1])
    i = 0
    # 낮은순의 캐릭터 체력 루프
    for health in healths:
        # 깎이는 체력이 낮은 순의 아이템 루프
        while i < len(items):
            # 착용할 수 없는 아이템이 나온다면 아이템 루프를 중단하고 다음 캐릭터로 넘어간다.
            if health - items[i][1] < 100:
                break
            # 착용할 수 있는 아이템이라면 heap에 넣는다. 한번 힙에 넣은 아이템은 다음 
            # 캐릭터에서 착용 여부를 확인할 필요가 없다.(반드시 다음엔 지금 캐릭터 
            # 보다  체력이 높은 캐릭터가 나오게 되므로 반드시 착용 가능하다.) min 
            # heap이 아니라 max heap으로 만들어야 되기 때문에(공격력이 높은순)
            # 공격력에 -를 붙여서 heap에 index와 함께 넣는다.
            heapq.heappush(heap, (-items[i][0], items[i][2]))
            i += 1
        # heap은 현재 캐릭터가 착용가능한 아이템들이 들어 있으며 여기서 pop을 하면, 
        # 공격력이 가장 높은 아이템이 나오게 된다.
        if heap:
            index = heapq.heappop(heap)[1]
            answer.append(index)
    answer.sort()
    return answer

```

heap을 활용해야 될 것 같다는 느낌이 있었지만, 위처럼 둘다 체력순이 아니라 items를 공격력 높은 순으로 정렬해 놓고 생각하다 보니, 힙을 쓸 부분을 찾지 못했고 결국 비효율을 유발해 채점할때 시간초가가 나왔다. 아쉬운 문제다.
