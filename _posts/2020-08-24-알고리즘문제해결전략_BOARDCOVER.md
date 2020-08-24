---
category: Algorithm
tags : [알고리즘문제해결전략, BOARDCOVER]
---

## BOARDCOVER 문제 개요

#### 문제
![example](http://algospot.com/media/judge-attachments/2b7bfee35cbec2f4e799bb011ac18f69/03.png "exmaple")  
H*W 크기의 게임판이 있습니다. 게임판은 검은 칸과 흰 칸으로 구성된 격자 모양을 하고 있는데 이 중 모든 흰 칸을 3칸짜리 L자 모양의 블록으로 덮고 싶습니다. 이 때 블록들은 자유롭게 회전해서 놓을 수 있지만, 서로 겹치거나, 검은 칸을 덮거나, 게임판 밖으로 나가서는 안 됩니다. 위 그림은 한 게임판과 이를 덮는 방법을 보여줍니다.

게임판이 주어질 때 이를 덮는 방법의 수를 계산하는 프로그램을 작성하세요.  
<br>

#### 입력
입력의 첫 줄에는 테스트 케이스의 수 C (C <= 30) 가 주어집니다. 각 테스트 케이스의 첫 줄에는 2개의 정수 H, W (1 <= H,W <= 20) 가 주어집니다. 다음 H 줄에 각 W 글자로 게임판의 모양이 주어집니다. # 은 검은 칸, . 는 흰 칸을 나타냅니다. 입력에 주어지는 게임판에 있는 흰 칸의 수는 50 을 넘지 않습니다.  
<br>

#### 출력
한 줄에 하나씩 흰 칸을 모두 덮는 방법의 수를 출력합니다.  
<br>

#### 예제 입력
3  
3 7  
#.....#  
#.....#  
##...##  
3 7  
#.....#  
#.....#  
##..###  
8 10  
##########  
#........#  
#........#  
#........#  
#........#  
#........#  
#........#  
##########  

#### 예제 출력
0  
2  
1514  

출처: [Algospot](https://algospot.com/judge/problem/read/BOARDCOVER)

## 구현

#### 설계
전에 구현한 PICNIC과 같이 교재에서 배운 것 처럼 재귀함수를 통해 구현을 하였다.

1. PICNIC과 마찬가지로 일정한 순서에 따라 하나의 포인트를 정하고 그 포인트에서 가능한 모든 경우에 대하여 블록을 끼울 수 있는지 찾아보고, 끼울 수 있는경우 그 블록을 끼운 상태로 하여 다시 나머지 빈자리들로 재귀함수를 돌린다. 이를 반복한다.
2. PICNIC과 다른점은 입력되는 데이터 수의 상한선이 없기 때문에, 동적으로 데이터 저장공간을 할당해 주어야한다는점.


#### 전체 코드


```cpp
#include <iostream>

using namespace std;

int TC, H, W;
int NumOfTotal, Answer, NumOfBlack;

int BoardCover(bool* covered, int ToCover, int H, int W)
{
    int ret = 0;
    NumOfTotal = H*W;

    //기저 사례 : 채워야할 블록이 없을 때, 모두 채워진 것이므로 1을 return한다.
    if(ToCover == 0)
        return 1;    
    //중심이 되는 블록 위치.
    int point1 = 0;
    int point2 = 0;

    //자료구조를 1차원 배열로 선언하고 값을 받았기 때문에, 블록의 가장 왼쪽위에서 오른쪽방향과 아래쪽 방향으로 빈 블록을 찾는다.
    for(int i = 0; i<NumOfTotal; i++)
    {
        if(covered[i] == 0)
        {
            point1 = i;
            break;
        }
    }

    //아래 한칸을 확인함
    if(point1 + W < NumOfTotal && covered[point1 + W] == 0)
    {
        //point2는 아래칸의 좌표값
        point2 = point1 + W;
        //확인한 아래칸의 왼쪽블럭을 확인함(1번 케이스)
        if(point2 % W - 1 >= 0 && covered[point2 - 1] == 0)
        {
            covered[point1] = 1;
            covered[point2] = 1;
            covered[point2 - 1] = 1;
            ret += BoardCover(covered, ToCover -3, H, W);
            covered[point1] = 0;
            covered[point2] = 0;
            covered[point2 - 1] = 0;
        }
        //확인한 아래칸의 오른쪽 블럭을 확인함.(2번 케이스)
        if(point2 % W + 1 < W && covered[point2 + 1] == 0)
        {
            covered[point1] = 1;
            covered[point2] = 1;
            covered[point2 + 1] = 1;
            ret += BoardCover(covered, ToCover -3, H, W);
            covered[point1] = 0;
            covered[point2] = 0;
            covered[point2 + 1] = 0;
        }
        //확인한 아래칸과 원래 중심 포인트의 오른쪽 블럭을 확인함.(3번 케이스)
        if(point1 % W + 1 < W && covered[point1 + 1] == 0)
        {
            covered[point1] = 1;
            covered[point2] = 1;
            covered[point1 + 1] = 1;
            ret += BoardCover(covered, ToCover -3, H, W);
            covered[point1] = 0;
            covered[point2] = 0;
            covered[point1 + 1] = 0;
        }
    }

    //중심 블럭의 오른쪽 블럭과 그 블럭의 아래쪽 블럭을 확인함.(4번 케이스)
    if(point1 % W + 1 < W && covered[point1 + 1] == 0)
    {
        if((point1 + 1) + W < NumOfTotal && covered[point1 + 1 + W] == 0)
        {
            covered[point1] = 1;
            covered[point1 + 1] = 1;
            covered[point1 + 1 + W] = 1;
            ret += BoardCover(covered, ToCover -3, H, W);
            covered[point1] = 0;
            covered[point1 + 1] = 0;
            covered[point1 + 1 + W] = 0;
        }
    }
    return ret;
}

int main(void)
{
    cin >> TC;
    for(int i = 0; i < TC; i++)
    {
        cin >> H >> W;
        NumOfTotal = H*W;
        NumOfBlack = 0;
        Answer = 0;
        bool* covered = new bool[NumOfTotal];
        
        for(int j = 0; j<NumOfTotal; j++)
            covered[j] = 0;
        char tmp;
        for(int j=0; j < NumOfTotal; j++)
        {
            cin >> tmp;
            if(tmp == '#')
            {
                covered[j] = 1;
                NumOfBlack++;
            }
        }
        if(NumOfBlack == NumOfTotal)
            Answer = 1;

        else if((NumOfTotal - NumOfBlack)%3 == 0)
            Answer = BoardCover(covered, NumOfTotal - NumOfBlack, H, W);

        cout<< Answer << endl;
    }
    return 0;
}
```
<br>

#### main함수
```cpp

int main(void)
{
    cin >> TC;
    for(int i = 0; i < TC; i++)
    {
        cin >> H >> W;
        NumOfTotal = H*W;
        NumOfBlack = 0;
        Answer = 0;
        bool* covered = new bool[NumOfTotal];
        
        for(int j = 0; j<NumOfTotal; j++)
            covered[j] = 0;
        char tmp;
        for(int j=0; j < NumOfTotal; j++)
        {
            cin >> tmp;
            if(tmp == '#')
            {
                covered[j] = 1;
                NumOfBlack++;
            }
        }
        if(NumOfBlack == NumOfTotal)
            Answer = 1;

        else if((NumOfTotal - NumOfBlack)%3 == 0)
            Answer = BoardCover(covered, NumOfTotal - NumOfBlack, H, W);

        cout<< Answer << endl;
    }
    return 0;
}
```
<br>

PICNIC의 main함수와 유사하게 구현하였다. Testcase룰 받고 그 수 만큼 함수를 실행시킨다. PICNIC과 다른점은 PICNIC의 경우 주어지는 데이터양의 상한선이 정해져 있어서 그 만큼만 할당해주면 되었지만, 이경우는 그렇지 않아서 입려받은 H와 W를 통해 배열의 크기를 바꾸어 주었다. 또한 전에 2차원배열을 사용했던 것과 달리 1차원 배열로 값들을 처리하였다.  

#### Recursive 함수(BoardCover)

```cpp
int BoardCover(bool* covered, int ToCover, int H, int W)
{
    int ret = 0;
    NumOfTotal = H*W;

    //기저 사례 : 채워야할 블록이 없을 때, 모두 채워진 것이므로 1을 return한다.
    if(ToCover == 0)
        return 1;    
    //중심이 되는 블록 위치.
    int point1 = 0;
    int point2 = 0;

    //자료구조를 1차원 배열로 선언하고 값을 받았기 때문에, 블록의 가장 왼쪽위에서 오른쪽방향과 아래쪽 방향으로 빈 블록을 찾는다.
    for(int i = 0; i<NumOfTotal; i++)
    {
        if(covered[i] == 0)
        {
            point1 = i;
            break;
        }
    }

    //아래 한칸을 확인함
    if(point1 + W < NumOfTotal && covered[point1 + W] == 0)
    {
        //point2는 아래칸의 좌표값
        point2 = point1 + W;
        //확인한 아래칸의 왼쪽블럭을 확인함(1번 케이스)
        if(point2 % W - 1 >= 0 && covered[point2 - 1] == 0)
        {
            covered[point1] = 1;
            covered[point2] = 1;
            covered[point2 - 1] = 1;
            ret += BoardCover(covered, ToCover -3, H, W);
            covered[point1] = 0;
            covered[point2] = 0;
            covered[point2 - 1] = 0;
        }
        //확인한 아래칸의 오른쪽 블럭을 확인함.(2번 케이스)
        if(point2 % W + 1 < W && covered[point2 + 1] == 0)
        {
            covered[point1] = 1;
            covered[point2] = 1;
            covered[point2 + 1] = 1;
            ret += BoardCover(covered, ToCover -3, H, W);
            covered[point1] = 0;
            covered[point2] = 0;
            covered[point2 + 1] = 0;
        }
        //확인한 아래칸과 원래 중심 포인트의 오른쪽 블럭을 확인함.(3번 케이스)
        if(point1 % W + 1 < W && covered[point1 + 1] == 0)
        {
            covered[point1] = 1;
            covered[point2] = 1;
            covered[point1 + 1] = 1;
            ret += BoardCover(covered, ToCover -3, H, W);
            covered[point1] = 0;
            covered[point2] = 0;
            covered[point1 + 1] = 0;
        }
    }

    //중심 블럭의 오른쪽 블럭과 그 블럭의 아래쪽 블럭을 확인함.(4번 케이스)
    if(point1 % W + 1 < W && covered[point1 + 1] == 0)
    {
        if((point1 + 1) + W < NumOfTotal && covered[point1 + 1 + W] == 0)
        {
            covered[point1] = 1;
            covered[point1 + 1] = 1;
            covered[point1 + 1 + W] = 1;
            ret += BoardCover(covered, ToCover -3, H, W);
            covered[point1] = 0;
            covered[point1 + 1] = 0;
            covered[point1 + 1 + W] = 0;
        }
    }
    return ret;
}
```

원래 위의 코드길이는 지금보다 2배정도 길었는데, 교재에 풀이를 보고 다음과 같이 바꾸게 되었다. 그 이유는 다음과 같은 실수를 했기 때문이다.  

처음에 구현할 때는 기준으로 잡은 포인트를 기준으로 다음의 12가지 경우를 생각했다. 

★-포인트 블럭  
O-확인하는 블럭  

O|O|X|:|X|O|O|:|X|O|X|:|X|O|X|
X|★|X|:|X|★|X|:|O|★|X|:|X|★|O|
X|X|X|:|X|X|X|:|X|X|X|:|X|X|X|
  
X|X|X|:|X|X|X|:|X|X|X|:|X|X|X|
X|★|X|:|X|★|X|:|O|★|X|:|X|★|O|
O|O|X|:|X|O|O|:|X|O|X|:|X|O|X|
  
O|X|X|:|X|X|X|:|X|X|O|:|X|X|X|
O|★|X|:|O|★|X|:|X|★|O|:|X|★|O|
X|X|X|:|O|X|X|:|X|X|X|:|X|X|O|
  
그래서 위의 12가지 방법을 전부 고려하는 방식으로 구현을 했다. 뭔가 줄일 수 있을거 같기도 했는데 그거 생각하는시간에 구현하는게 낫겠다 싶어서 그냥 했는데, 대단히 바보같은 짓이었다. 위에서 포인트 블럭을 찾는 과정을 생각해 보면 책을 읽는 것처럼 왼쪽에서 오른쪽으로 한 행을 차례로 읽고 그 후에 다음행을 읽는 식이다. 따라서, 포인트 블럭이 정해지면 그 블록의 왼쪽 블록들과 위의 행들의 블럭들은 반드시 1이 된다. 그러므로 확인해보아야 하는 경우의 수는 다음의 네가지 밖에 남지 않는다. 각각을 차례로 1번, 2번, 3번, 4번 케이스라고 하였을때의 구현이 위와같다.  

X|X|X|:|X|X|X|:|X|X|X|:|X|X|X|
X|★|X|:|X|★|X|:|X|★|O|:|X|★|O|
O|O|X|:|X|O|O|:|X|O|X|:|X|X|O|
  
<br>


## 정리
1. Global변수 H,W를 선언했음에도, 위에서 구현할때는 함수의 매개변수로 계속 념겨주었다. 저번에 PICNIC을 구현할 때, 배열을 글로벌 변수로 선언했음에도 불구하고 함수를 부를때마다 이유없이 다른 값들이 입력되어서 이번에는 처음부터 매개 변수로 넘겨주었다. 당연히 지금까지 배운것과는 다른 방식이었다. 구현을 끝낸 후에 원래대로 매개변수로 넘기지 않고 글로벌 변수들을 사용했는데 아무 문제 없었다. PICNIC의 경우를 다시 살펴봐야 할 것 같다.

2. if(point1 % W + 1 < W && covered[point1 + 1] == 0) 과 같이 하나의 if문 안에서 &&나 ||로 두가지 이상을 확인하는경우 만약에 둘을 동시에 확인한다면 전자의 조건이 충족 되지 않았을 때(게임판의 범위를 벗어나는경우), 뒤의 경우를 확인하면 분명히 Segmentation fault를 발생시킬 것이라고 생각했다. 그런데 if문에서 지금처럼 조건이 여러개인 경우 첫번째가 위반되는 경우 뒤를 확인하지 않는다고 얼핏 배웠던 것 같아서 그대로 구현을 하였다. 구현을 띁내고 알아보니 배웠던 것이 맞았다. 이 기회에 다시 상기시켰다.

PICNIC보다 내용이 쉽지는 않았던 것 같지만, C++구현이 좀 더 익숙해져서 전보다 빨리 끝냈던 것 같다. 처음에는 코드를 너무 길게 짜서 디버깅하기가 버거웠다. 사실 완전히 잘못 구현한 부분도 있었는데 다른 테스트 케이스를 통해 해결했다.(너무 바보같은 실수여서 안씀 ㅋ)

끝!