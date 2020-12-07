---
category: programmers
tags: [K-digital training, week1_day1]
---

## LinkedList PopAt() 구현

처음으로 시간이 걸리는 과제가 나왔다. 계속 실패가 뜨길래 직접 코드를 복사해서 돌려보다가 답을 찾게 되었다 원래 코드는 다음과 같았다.   

```python
def popAt(self, pos):
        if pos < 1 or pos > self.nodeCount:
            raise IndexError
        if pos == 1:
            data = self.head.data
            self.head = self.head.next
            self.nodeCount -= 1
            return data
        else:
            curr = self.getAt(pos)
            prev = self.getAt(pos - 1)
            data = curr.data
            prev.next = curr.next
            self.nodeCount -= 1
            return data
```

무작정 작성했을때는 문제가 없는 것 처럼 보였는데 직접 돌려보니까 에러가 발생했고 주로 맨 마지막 노드를 삭제할때 문제가 있었다. 그 원인은 마지막 노드를 삭제할 때, tail노드를 지정해주지 않아서 였다. 이를 수정하여 다음과 같이 바꾸어 주었다.   

```python
def popAt(self, pos):
        if pos < 1 or pos > self.nodeCount:
            raise IndexError
        if pos == 1:
            data = self.head.data
            self.head = self.head.next
            if self.head is None:
                self.tail = None
            self.nodeCount -= 1
            return data
        else:
            curr = self.getAt(pos)
            prev = self.getAt(pos - 1)
            data = curr.data
            prev.next = curr.next
            if prev.next is None:
                self.tail = prev
            self.nodeCount -= 1
            return data

```

## DoublyLinkedList

DoublyLinkedList를 순회할 때의 코드는 아래와 같다.
```python
    def reverse(self):
        tmp = []
        curr = self.tail
        while curr.prev.prev:
            tmp.append(curr.prev.data)
            curr = curr.prev
        return tmp
```
while문을 curr.prev.prev가 아닌 curr.prev로 하면 마지막에 있는 더미 노드의 값(None)까지 읽어드리게 된다.   
